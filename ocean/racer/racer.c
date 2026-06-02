/**
 * WhiskerRacer + Car Physics Integration
 *
 * Original WhiskerRacer environment by Omar.
 * Car dynamics ported from Python/Box2D to C (originally by Oleg Klimov,
 * see http://www.iforce2d.net/b2dtut/top-down-car).
 *
 * The Box2D rigid-body simulation has been replaced with a lightweight,
 * self-contained 2-D rigid-body integrator so that the file has no external
 * physics dependency beyond raylib and the C standard library.
 */
 
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <stdbool.h>
#include "raylib.h"
#include <time.h>
 
/* -------------------------------------------------------------------------
 * Action constants
 * ---------------------------------------------------------------------- */
#define LEFT  0
#define NOOP  1
#define RIGHT 2
 
/* -------------------------------------------------------------------------
 * Math helpers
 * ---------------------------------------------------------------------- */
#define PI2 (PI * 2.0f)
 
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float signf_int(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }
 
/* -------------------------------------------------------------------------
 * Track geometry constants
 * ---------------------------------------------------------------------- */
#define MAX_CONTROL_POINTS    32
#define NUM_RADIAL_SECTORS    16
#define MAX_BEZIER_RESOLUTION 16
 
/* -------------------------------------------------------------------------
 * Car physics constants  (ported 1-to-1 from car_dynamics.py)
 * ---------------------------------------------------------------------- */
#define CAR_SIZE                0.02f
#define ENGINE_POWER            (100000000.0f * CAR_SIZE * CAR_SIZE)
#define WHEEL_MOMENT_OF_INERTIA (4000.0f      * CAR_SIZE * CAR_SIZE)
#define FRICTION_LIMIT          (1000000.0f   * CAR_SIZE * CAR_SIZE)
 
#define WHEEL_R  27
#define WHEEL_W  14
 
/* Wheel positions relative to hull centre (design units, pre-SIZE scale) */
static const float WHEELPOS[4][2] = {
    {-55.0f, +80.0f},   /* front-left  */
    {+55.0f, +80.0f},   /* front-right */
    {-55.0f, -82.0f},   /* rear-left   */
    {+55.0f, -82.0f},   /* rear-right  */
};
 
/* Colours */
static const Color WHEEL_COLOR_RL = {  0,   0,   0, 255};
static const Color WHEEL_WHITE_RL = { 77,  77,  77, 255};
static const Color MUD_COLOR_RL   = {102, 102,   0, 255};
static const Color HULL_COLOR_RL  = {204,   0,   0, 255};   /* ~(0.8,0,0) */
 
/* -------------------------------------------------------------------------
 * Lightweight 2-D rigid body  (replaces Box2D b2Body)
 * ---------------------------------------------------------------------- */
typedef struct {
    float x, y;          /* world position */
    float angle;         /* radians */
    float vx, vy;        /* linear velocity */
    float omega_body;    /* angular velocity of the body itself */
    float mass;          /* kg */
    float inv_mass;
    float inertia;
    float inv_inertia;
} RigidBody;
 
static void rb_apply_force_to_center(RigidBody *b, float fx, float fy, float dt) {
    b->vx += fx * b->inv_mass * dt;
    b->vy += fy * b->inv_mass * dt;
}
 
/* Return the world-space direction of a local vector (rotated by body angle) */
static void rb_get_world_vector(const RigidBody *b, float lx, float ly,
                                float *wx, float *wy) {
    float c = cosf(b->angle), s = sinf(b->angle);
    *wx = c * lx - s * ly;
    *wy = s * lx + c * ly;
}
 
/* -------------------------------------------------------------------------
 * Revolute joint (simplified: drives motor speed toward target) ported from
 * Box2D revoluteJointDef – we only need the steer angle + motor logic.
 * ---------------------------------------------------------------------- */
typedef struct {
    float angle;        /* current joint angle   */
    float motor_speed;  /* target motor speed    */
    float lower_angle;  /* -0.4 rad              */
    float upper_angle;  /* +0.4 rad              */
} RevoluteJoint;
 
/* -------------------------------------------------------------------------
 * Skid / mud particle  (matches Python Particle class)
 * ---------------------------------------------------------------------- */
#define MAX_SKID_POLY    30
#define MAX_PARTICLES    30
 
typedef struct {
    float  poly[MAX_SKID_POLY][2];
    int    poly_count;
    Color  color;
    float  ttl;
    bool   grass;
} Particle;
 
/* -------------------------------------------------------------------------
 * Single wheel state
 * ---------------------------------------------------------------------- */
typedef struct {
    RigidBody   body;
    RevoluteJoint joint;
 
    float wheel_rad;    /* physical radius in metres */
    Color color;
 
    float gas;
    float brake;
    float steer;        /* target steer angle */
    float phase;        /* rolling phase (for white stripe) */
    float omega;        /* spin angular velocity            */
 
    bool  skid_start_valid;
    float skid_start[2];
    Particle *skid_particle; /* pointer into car->particles, or NULL */
 
    /* tiles – replaced by a simple boolean: is the wheel on road? */
    float road_friction; /* 1.0 = tarmac, 0.6 = grass */
} Wheel;
 
/* -------------------------------------------------------------------------
 * Car (hull + 4 wheels + particle list)
 * ---------------------------------------------------------------------- */
#define NUM_WHEELS 4
 
typedef struct {
    RigidBody hull;
    Wheel     wheels[NUM_WHEELS];
    float     fuel_spent;
 
    Particle  particles[MAX_PARTICLES];
    int       particle_count;
} Car;
 
/* Forward declaration */
static Particle *car_create_particle(Car *car,
                                     float x1, float y1,
                                     float x2, float y2,
                                     bool grass);
 
/* --- car_init -------------------------------------------------------------
 * Initialise car at world position (init_x, init_y) facing init_angle.
 * ---------------------------------------------------------------------- */
static void car_init(Car *car, float init_angle, float init_x, float init_y) {
    memset(car, 0, sizeof(Car));
 
    /* Hull */
    car->hull.x     = init_x;
    car->hull.y     = init_y;
    car->hull.angle = init_angle;
    car->hull.mass  = 4.0f;   /* approximate – density=1 over 4 polys */
    car->hull.inv_mass    = 1.0f / car->hull.mass;
    car->hull.inertia     = 0.5f * car->hull.mass * (0.5f * 0.5f);
    car->hull.inv_inertia = 1.0f / car->hull.inertia;
 
    /* Wheels */
    for (int i = 0; i < NUM_WHEELS; i++) {
        Wheel *w = &car->wheels[i];
        w->body.x     = init_x  + WHEELPOS[i][0] * CAR_SIZE;
        w->body.y     = init_y  + WHEELPOS[i][1] * CAR_SIZE;
        w->body.angle = init_angle;
        w->body.mass  = 0.1f;
        w->body.inv_mass    = 1.0f / w->body.mass;
        w->body.inertia     = 0.01f;
        w->body.inv_inertia = 1.0f / w->body.inertia;
 
        w->wheel_rad = WHEEL_R * CAR_SIZE;
        w->color     = WHEEL_COLOR_RL;
 
        /* Joint limits */
        w->joint.lower_angle = -0.4f;
        w->joint.upper_angle = +0.4f;
 
        /* Initial road state */
        w->road_friction = 1.0f;
    }
}
 
/* --- car_gas -------------------------------------------------------------
 * Apply throttle to rear wheels (indices 2 & 3).  gas in [0,1].
 * ---------------------------------------------------------------------- */
static void car_gas(Car *car, float gas) {
    gas = clampf(gas, 0.0f, 1.0f);
    for (int i = 2; i < 4; i++) {
        float diff = gas - car->wheels[i].gas;
        if (diff > 0.1f) diff = 0.1f;   /* gradual increase only */
        car->wheels[i].gas += diff;
    }
}
 
/* --- car_brake ------------------------------------------------------------
 * b in [0,1].  > 0.9 locks wheels.
 * ---------------------------------------------------------------------- */
static void car_brake(Car *car, float b) {
    for (int i = 0; i < NUM_WHEELS; i++)
        car->wheels[i].brake = b;
}
 
/* --- car_steer ------------------------------------------------------------
 * s in [-1,1].  Affects front wheels (0 & 1).
 * ---------------------------------------------------------------------- */
static void car_steer(Car *car, float s) {
    car->wheels[0].steer = s;
    car->wheels[1].steer = s;
}
 
/* --- car_step ------------------------------------------------------------
 * Advance physics by dt seconds.
 * ---------------------------------------------------------------------- */
static void car_step(Car *car, float dt) {
    for (int i = 0; i < NUM_WHEELS; i++) {
        Wheel    *w  = &car->wheels[i];
        RigidBody *b = &w->body;
 
        /* --- Steer motor --- */
        float dir = signf_int(w->steer - w->joint.angle);
        float val = fabsf(w->steer - w->joint.angle);
        w->joint.motor_speed = dir * fminf(50.0f * val, 3.0f);
 
        /* Integrate joint angle, clamped to limits */
        w->joint.angle += w->joint.motor_speed * dt;
        w->joint.angle  = clampf(w->joint.angle,
                                 w->joint.lower_angle,
                                 w->joint.upper_angle);
 
        /* Wheel body angle = hull angle + joint angle */
        b->angle = car->hull.angle + w->joint.angle;
 
        /* --- Friction limit --- */
        float friction_limit = FRICTION_LIMIT * w->road_friction;
 
        /* --- World vectors for wheel forward/side --- */
        float forw_x, forw_y, side_x, side_y;
        rb_get_world_vector(b, 0.0f, 1.0f, &forw_x, &forw_y);
        rb_get_world_vector(b, 1.0f, 0.0f, &side_x, &side_y);
 
        float vf = forw_x * b->vx + forw_y * b->vy;  /* forward speed */
        float vs = side_x * b->vx + side_y * b->vy;  /* side speed    */
 
        /* --- Engine / spin --- */
        w->omega += dt * ENGINE_POWER * w->gas
                    / WHEEL_MOMENT_OF_INERTIA
                    / (fabsf(w->omega) + 5.0f);
        car->fuel_spent += dt * ENGINE_POWER * w->gas;
 
        /* --- Braking --- */
        if (w->brake >= 0.9f) {
            w->omega = 0.0f;
        } else if (w->brake > 0.0f) {
            const float BRAKE_FORCE = 15.0f;
            float bdir = -signf_int(w->omega);
            float bval = BRAKE_FORCE * w->brake;
            if (fabsf(bval) > fabsf(w->omega)) bval = fabsf(w->omega);
            w->omega += bdir * bval;
        }
        w->phase += w->omega * dt;
 
        /* --- Force computation --- */
        float vr      = w->omega * w->wheel_rad;
        float f_force = (-vf + vr) * 205000.0f * CAR_SIZE * CAR_SIZE;
        float p_force = (-vs)      * 205000.0f * CAR_SIZE * CAR_SIZE;
        float force   = sqrtf(f_force * f_force + p_force * p_force);
 
        /* --- Skid trace --- */
        bool grass = (w->road_friction < 0.8f);
        if (force > 2.0f * friction_limit) {
            if (w->skid_particle != NULL
                && w->skid_particle->grass == grass
                && w->skid_particle->poly_count < MAX_SKID_POLY) {
                int idx = w->skid_particle->poly_count;
                w->skid_particle->poly[idx][0] = b->x;
                w->skid_particle->poly[idx][1] = b->y;
                w->skid_particle->poly_count++;
            } else if (!w->skid_start_valid) {
                w->skid_start[0]    = b->x;
                w->skid_start[1]    = b->y;
                w->skid_start_valid = true;
            } else {
                w->skid_particle = car_create_particle(
                    car,
                    w->skid_start[0], w->skid_start[1],
                    b->x,             b->y,
                    grass);
                w->skid_start_valid = false;
            }
        } else {
            w->skid_start_valid = false;
            w->skid_particle    = NULL;
        }
 
        /* --- Clamp force to friction limit --- */
        if (force > friction_limit) {
            f_force = f_force / force * friction_limit;
            p_force = p_force / force * friction_limit;
        }
 
        /* --- Feed back into wheel spin --- */
        w->omega -= dt * f_force * w->wheel_rad / WHEEL_MOMENT_OF_INERTIA;
 
        /* --- Apply force to wheel body --- */
        float apply_x = p_force * side_x + f_force * forw_x;
        float apply_y = p_force * side_y + f_force * forw_y;
 
        b->vx += apply_x * b->inv_mass * dt;
        b->vy += apply_y * b->inv_mass * dt;
 
        /* --- Integrate wheel body position --- */
        b->x += b->vx * dt;
        b->y += b->vy * dt;
    }
 
    /* --- Integrate hull from average wheel forces ---
     * (In the original, Box2D propagates forces through joints automatically.
     *  Here we simply average the four wheel velocities to drive the hull,
     *  preserving the kinematic feel without a full constraint solver.) */
    float avg_vx = 0.0f, avg_vy = 0.0f;
    for (int i = 0; i < NUM_WHEELS; i++) {
        avg_vx += car->wheels[i].body.vx;
        avg_vy += car->wheels[i].body.vy;
    }
    car->hull.vx = avg_vx * 0.25f;
    car->hull.vy = avg_vy * 0.25f;
    car->hull.x += car->hull.vx * dt;
    car->hull.y += car->hull.vy * dt;
 
    /* Update wheel body world positions to stay attached to hull */
    for (int i = 0; i < NUM_WHEELS; i++) {
        Wheel *w = &car->wheels[i];
        float c  = cosf(car->hull.angle);
        float s  = sinf(car->hull.angle);
        float lx = WHEELPOS[i][0] * CAR_SIZE;
        float ly = WHEELPOS[i][1] * CAR_SIZE;
        w->body.x = car->hull.x + c * lx - s * ly;
        w->body.y = car->hull.y + s * lx + c * ly;
    }
}
 
/* --- car_create_particle --------------------------------------------------
 * Mimics Python Car._create_particle.
 * ---------------------------------------------------------------------- */
static Particle *car_create_particle(Car *car,
                                     float x1, float y1,
                                     float x2, float y2,
                                     bool grass) {
    /* Evict oldest if full */
    if (car->particle_count == MAX_PARTICLES) {
        memmove(&car->particles[0], &car->particles[1],
                sizeof(Particle) * (MAX_PARTICLES - 1));
        car->particle_count--;
    }
    Particle *p = &car->particles[car->particle_count++];
    p->color       = grass ? MUD_COLOR_RL : WHEEL_COLOR_RL;
    p->ttl         = 1.0f;
    p->grass        = grass;
    p->poly[0][0]  = x1;  p->poly[0][1] = y1;
    p->poly[1][0]  = x2;  p->poly[1][1] = y2;
    p->poly_count  = 2;
 
    /* Invalidate stale skid_particle pointers in all wheels */
    for (int i = 0; i < NUM_WHEELS; i++) {
        if (car->wheels[i].skid_particle != NULL) {
            /* Check if it still points into a valid slot */
            ptrdiff_t off = car->wheels[i].skid_particle - car->particles;
            if (off < 0 || off >= car->particle_count)
                car->wheels[i].skid_particle = NULL;
        }
    }
    return p;
}
 
/* --- car_draw -------------------------------------------------------------
 * Renders the car using raylib DrawPoly.
 * zoom / translation / angle mimic the Python draw() signature.
 * screen_y_flip = env->height used to convert from physics Y-up to screen Y-down.
 * ---------------------------------------------------------------------- */
static void car_draw(const Car *car, float zoom, float tx, float ty,
                     float view_angle, int screen_h, bool draw_particles) {
 
    /* Helper: transform a world point → screen point */
#define WORLD_TO_SCREEN(wx, wy, sx, sy)                           \
    do {                                                           \
        float _rx = (wx) * cosf(view_angle) - (wy) * sinf(view_angle); \
        float _ry = (wx) * sinf(view_angle) + (wy) * cosf(view_angle); \
        (sx) = _rx * zoom + tx;                                    \
        (sy) = _ry * zoom + ty;                                    \
    } while (0)
 
    if (draw_particles) {
        for (int pi = 0; pi < car->particle_count; pi++) {
            const Particle *p = &car->particles[pi];
            if (p->poly_count < 2) continue;
            for (int j = 0; j < p->poly_count - 1; j++) {
                float x1s, y1s, x2s, y2s;
                WORLD_TO_SCREEN(p->poly[j][0],   p->poly[j][1],   x1s, y1s);
                WORLD_TO_SCREEN(p->poly[j+1][0], p->poly[j+1][1], x2s, y2s);
                DrawLineEx((Vector2){x1s, y1s}, (Vector2){x2s, y2s}, 2.0f, p->color);
            }
        }
    }
 
    /* Draw hull as a filled quad (use HULL_POLY3 bounding box approximation) */
    {
        float hw = 50.0f * CAR_SIZE * zoom;
        float hh = 90.0f * CAR_SIZE * zoom;
        float c = cosf(car->hull.angle + view_angle);
        float s = sinf(car->hull.angle + view_angle);
        float cx, cy;
        WORLD_TO_SCREEN(car->hull.x, car->hull.y, cx, cy);
        DrawRectanglePro(
            (Rectangle){ cx, cy, hw * 2.0f, hh * 2.0f },
            (Vector2){ hw, hh },
            -(car->hull.angle + view_angle) * (180.0f / PI),
            HULL_COLOR_RL
        );
    }
 
    /* Draw wheels */
    for (int i = 0; i < NUM_WHEELS; i++) {
        const Wheel *w = &car->wheels[i];
        float ww = WHEEL_W * CAR_SIZE * zoom;
        float wr = WHEEL_R * CAR_SIZE * zoom;
        float cx, cy;
        WORLD_TO_SCREEN(w->body.x, w->body.y, cx, cy);
 
        DrawRectanglePro(
            (Rectangle){ cx, cy, ww * 2.0f, wr * 2.0f },
            (Vector2){ ww, wr },
            -(w->body.angle + view_angle) * (180.0f / PI),
            WHEEL_COLOR_RL
        );
 
        /* White stripe (phase-based, same logic as Python) */
        float a1 = w->phase;
        float a2 = w->phase + 1.2f;
        float s1 = sinf(a1), s2 = sinf(a2);
        float c1 = cosf(a1), c2 = cosf(a2);
        if (!(s1 > 0.0f && s2 > 0.0f)) {
            if (s1 > 0.0f) c1 = signf_int(c1);
            if (s2 > 0.0f) c2 = signf_int(c2);
            /* Draw a thin white quad inside the wheel rect */
            float stripe_y1 = WHEEL_R * c1 * CAR_SIZE * zoom;
            float stripe_y2 = WHEEL_R * c2 * CAR_SIZE * zoom;
            float sx1, sy1, sx2, sy2;
            WORLD_TO_SCREEN(w->body.x, w->body.y + stripe_y1, sx1, sy1);
            WORLD_TO_SCREEN(w->body.x, w->body.y + stripe_y2, sx2, sy2);
            DrawLineEx((Vector2){sx1 - ww, sy1},
                       (Vector2){sx1 + ww, sy2},
                       2.0f, WHEEL_WHITE_RL);
        }
    }
#undef WORLD_TO_SCREEN
}
 
/* =========================================================================
 * Track geometry structures (unchanged from original)
 * ====================================================================== */
typedef struct {
    Vector2 position;
} ControlPoint;
 
typedef struct {
    ControlPoint controls[MAX_CONTROL_POINTS];
    int num_points;
    Vector2 centerline[MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 inner_edge [MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    Vector2 outer_edge [MAX_CONTROL_POINTS * MAX_BEZIER_RESOLUTION];
    int total_points;
    Vector2 curbs[MAX_CONTROL_POINTS][4];
    int curb_count;
} Track;
 
/* =========================================================================
 * Logging / Client structs (unchanged)
 * ====================================================================== */
typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;
 
typedef struct Client {
    float width;
    float height;
    float flw_ang;
    float frw_ang;
    float max_whisker_length;
    float turn_pi_frac;
    float maxv;
    int   render;
    int   debug;
} Client;
 
/* =========================================================================
 * WhiskerRacer environment
 * ====================================================================== */
typedef struct WhiskerRacer {
    Client *client;
    Log     log;
    float  *observations;
    float  *actions;
    float  *rewards;
    float  *terminals;
    int     num_agents;
    int     i;
 
    int          debug;
    unsigned int rng;
    int          render_many;
 
    float corner_thresh;
    float ftmp1, ftmp2, ftmp3, ftmp4;
    int   method;
 
    float reward_yellow;
    float reward_green;
    float gamma;
 
    /* Game */
    int   width;
    int   height;
    float score;
    int   tick;
    int   max_score;
    int   half_max_score;
    int   frameskip;
    int   render;
    int   continuous;
    int   current_sector;
    int   sectors_completed[NUM_RADIAL_SECTORS];
    int   total_sectors_crossed;
    int   track_width;
    int   num_radial_sectors;
    int   num_points;
    int   bezier_resolution;
    Track track;
 
    /* Car (replaces bare px/py/vx/vy with full physics car) */
    Car   car;                /* <-- NEW: full car state */
 
    /* Legacy scalar position/angle kept in sync with car.hull for backward
     * compatibility with whisker and radial-progress code. */
    float px, py, ang, vx, vy, v;
    int   near_point_idx;
    float maxv;
    float turn_pi_frac;
 
    /* Whiskers */
    int     num_whiskers;
    Vector2 whisker_dirs[2];
    float   w_ang;
    float   llw_ang, flw_ang, frw_ang, rrw_ang;
    float   llw_length, flw_length, ffw_length, frw_length, rrw_length;
    float   max_whisker_length;
 
    /* Cached reciprocals */
    float inv_width, inv_height, inv_maxv, inv_pi2, inv_bezier_res;
 
    Texture2D puffer;
    int       texture_initialized;
    int       mode7;
} WhiskerRacer;
 
/* =========================================================================
 * Utility: sync legacy scalars ↔ car hull
 * ====================================================================== */
static void sync_car_to_env(WhiskerRacer *env) {
    env->px  = env->car.hull.x;
    env->py  = env->car.hull.y;
    env->ang = env->car.hull.angle;
    env->vx  = env->car.hull.vx;
    env->vy  = env->car.hull.vy;
}
 
static void sync_env_to_car(WhiskerRacer *env) {
    env->car.hull.x     = env->px;
    env->car.hull.y     = env->py;
    env->car.hull.angle = env->ang;
}
 
/* =========================================================================
 * Forward declarations (track helpers – implementations unchanged below)
 * ====================================================================== */
static void GenerateRandomTrack(WhiskerRacer *env);
 
/* =========================================================================
 * Cleanup / allocation helpers
 * ====================================================================== */
void c_close(WhiskerRacer *env) { (void)env; }
 
void free_allocated(WhiskerRacer *env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}
 
/* =========================================================================
 * Logging
 * ====================================================================== */
static void add_log(WhiskerRacer *env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score          += env->score;
    env->log.perf           += env->score / (float)env->max_score;
    env->log.n              += 1.0f;
}
 
/* =========================================================================
 * Observations
 * ====================================================================== */
static void compute_observations(WhiskerRacer *env) {
    env->observations[0] = env->flw_length;
    env->observations[1] = env->frw_length;
    env->observations[2] = env->score / 100.0f;
}
 
/* =========================================================================
 * Client / window
 * ====================================================================== */
Client *make_client(WhiskerRacer *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width             = (float)env->width;
    client->height            = (float)env->height;
    client->flw_ang           = env->flw_ang;
    client->frw_ang           = env->frw_ang;
    client->max_whisker_length = env->max_whisker_length;
    client->turn_pi_frac      = env->turn_pi_frac;
    client->maxv              = env->maxv;
 
    InitWindow(env->width, env->height, "Omar's AI Racer");
    SetTargetFPS(env->render_many ? 10 / env->frameskip : 60 / env->frameskip);
    env->puffer = LoadTexture("resources/shared/f1.png");
    return client;
}
 
void close_client(Client *client) {
    CloseWindow();
    free(client);
}
 
/* =========================================================================
 * Reset helpers
 * ====================================================================== */
static void get_random_start(WhiskerRacer *env) {
    int start_idx      = rand() % env->track.total_points;
    env->near_point_idx = start_idx;
 
    env->px = env->track.centerline[start_idx].x;
    env->py = env->track.centerline[start_idx].y;
 
    int next_idx = (start_idx + 1) % env->track.total_points;
    float dx = env->track.centerline[next_idx].x - env->px;
    float dy = env->track.centerline[next_idx].y - env->py;
    env->ang = atan2f(dy, dx);
 
    env->whisker_dirs[0] = (Vector2){cosf(env->ang + env->flw_ang),
                                     sinf(env->ang + env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang + env->frw_ang),
                                     sinf(env->ang + env->frw_ang)};
 
    env->v = env->maxv;
    env->flw_length = 0.50f;
    env->frw_length = 0.50f;
 
    /* Initialise car physics at the same spawn point */
    car_init(&env->car, env->ang, env->px, env->py);
    /* Give rear wheels some initial speed to match env->v */
    for (int i = 2; i < NUM_WHEELS; i++)
        env->car.wheels[i].omega = env->v / env->car.wheels[i].wheel_rad;
}
 
static void reset_radial_progress(WhiskerRacer *env) {
    float center_x = env->width  * 0.5f;
    float center_y = env->height * 0.5f;
    float angle = atan2f(env->py - center_y, env->px - center_x);
    if (angle < 0.0f) angle += PI2;
    env->current_sector = (int)(angle / (PI2 / 16.0f)) % 16;
    for (int i = 0; i < 16; i++) env->sectors_completed[i] = 0;
    env->total_sectors_crossed = 0;
}
 
static void reset_round(WhiskerRacer *env) {
    get_random_start(env);
    reset_radial_progress(env);
    env->vx = 0.0f;
    env->vy = 0.0f;
    env->v  = env->maxv;
}
 
void c_reset(WhiskerRacer *env) {
    compute_observations(env);
    env->score = 0.0f;
    reset_round(env);
    env->tick = 0;
}
 
/* =========================================================================
 * Line-segment intersection helper (unchanged)
 * ====================================================================== */
static inline int line_segment_intersect(Vector2 ray_start, Vector2 ray_dir,
                                         float ray_length,
                                         Vector2 seg_start, Vector2 seg_end,
                                         float *t_out) {
    Vector2 seg_dir = {seg_end.x - seg_start.x, seg_end.y - seg_start.y};
    Vector2 diff    = {seg_start.x - ray_start.x, seg_start.y - ray_start.y};
 
    float cross_rd_sd   = ray_dir.x * seg_dir.y - ray_dir.y * seg_dir.x;
    if (fabsf(cross_rd_sd) < 1e-3f) return 0;
 
    float cross_diff_sd = diff.x * seg_dir.y - diff.y * seg_dir.x;
    float cross_diff_rd = diff.x * ray_dir.y  - diff.y * ray_dir.x;
 
    float t = cross_diff_sd / cross_rd_sd;
    float u = cross_diff_rd / cross_rd_sd;
 
    if (t >= 0.0f && t <= ray_length && u >= 0.0f && u <= 1.0f) {
        *t_out = t;
        return 1;
    }
    return 0;
}
 
/* =========================================================================
 * Nearest track point
 * ====================================================================== */
static void update_nearest_point(WhiskerRacer *env) {
    float min_dist_sq = 100000.0f;
    int   closest_seg = env->near_point_idx;
    Vector2 car_pos   = {env->px, env->py};
 
    for (int offset = 0; offset <= 3; offset++) {
        int i = (env->near_point_idx + offset + env->track.total_points)
                % env->track.total_points;
        float dx = car_pos.x - env->track.centerline[i].x;
        float dy = car_pos.y - env->track.centerline[i].y;
        float d2 = dx * dx + dy * dy;
        if (d2 < min_dist_sq) { min_dist_sq = d2; closest_seg = i; }
    }
    env->near_point_idx = closest_seg;
}
 
/* =========================================================================
 * Whisker length calculation + update road friction on car wheels
 * ====================================================================== */
static void calc_whisker_lengths(WhiskerRacer *env) {
    float max_len     = env->max_whisker_length;
    float inv_max_len = 1.0f / max_len;
 
    update_nearest_point(env);
 
    float *lengths[2] = {&env->flw_length, &env->frw_length};
    Vector2 car_pos   = {env->px, env->py};
 
    for (int w = 0; w < 2; w++) {
        Vector2 whisker_dir   = env->whisker_dirs[w];
        float   min_hit       = max_len;
        int     window_size   = 10;
 
        for (int offset = -window_size / 2; offset <= window_size / 2; offset++) {
            int i      = (env->near_point_idx + offset
                          + env->track.total_points) % env->track.total_points;
            int next_i = (i + 1) % env->track.total_points;
            float t;
 
            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                       env->track.inner_edge[i],
                                       env->track.inner_edge[next_i], &t)) {
                if (t < min_hit) min_hit = t;
                if (t < 0.05f) break;
            }
            if (line_segment_intersect(car_pos, whisker_dir, max_len,
                                       env->track.outer_edge[i],
                                       env->track.outer_edge[next_i], &t)) {
                if (t < min_hit) min_hit = t;
                if (t < 0.05f) break;
            }
        }
 
        *lengths[w] = clampf(min_hit * inv_max_len, 0.0f, 1.0f);
 
        if (*lengths[w] < 0.05f) {   /* crashed */
            for (int j = 0; j < 2; j++) *lengths[j] = 0.0f;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            return;
        }
    }
 
    if (*lengths[0] >= 0.99f && *lengths[1] >= 0.99f) {  /* left track */
        for (int j = 0; j < 2; j++) *lengths[j] = 0.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
 
    /* Update road friction for car wheels based on whether they're on tarmac */
    for (int wi = 0; wi < NUM_WHEELS; wi++) {
        /* Simple approximation: if both whiskers near full length → on grass */
        float near = fminf(env->flw_length, env->frw_length);
        env->car.wheels[wi].road_friction = (near > 0.1f) ? 1.0f : 0.6f;
    }
}
 
/* =========================================================================
 * Radial progress
 * ====================================================================== */
static void update_radial_progress(WhiskerRacer *env) {
    float center_x = env->width  * 0.5f;
    float center_y = env->height * 0.5f;
 
    float angle = atan2f(env->py - center_y, env->px - center_x);
    if (angle < 0.0f) angle += PI2;
 
    int sector = (int)(angle / (PI2 / 16.0f)) % env->num_radial_sectors;
 
    if (sector != env->current_sector) {
        int expected_next = (env->current_sector + 1) % 16;
        if (sector == expected_next) {
            if (!env->sectors_completed[sector]) {
                env->sectors_completed[sector] = 1;
                env->total_sectors_crossed++;
                env->rewards[0] += env->reward_yellow;
                env->score      += env->reward_yellow;
            } else {
                env->rewards[0] += env->reward_yellow;
                env->score      += env->reward_yellow;
            }
        }
        env->current_sector = sector;
    }
}
 
/* =========================================================================
 * Bezier helpers (unchanged)
 * ====================================================================== */
static Vector2 EvaluateCubicBezier(Vector2 p0, Vector2 p1,
                                   Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float tt = t * t, uu = u * u;
    float uuu = uu * u, ttt = tt * t;
    return (Vector2){
        uuu * p0.x + 3*uu*t*p1.x + 3*u*tt*p2.x + ttt*p3.x,
        uuu * p0.y + 3*uu*t*p1.y + 3*u*tt*p2.y + ttt*p3.y
    };
}
 
static Vector2 NormalizeVector(Vector2 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y);
    if (len < 1e-5f) return (Vector2){0,0};
    return (Vector2){v.x/len, v.y/len};
}
 
static Vector2 GetPerpendicular(Vector2 v) { return (Vector2){-v.y, v.x}; }
 
/* =========================================================================
 * Track generation (unchanged from original)
 * ====================================================================== */
static void GenerateRandomControlPoints(WhiskerRacer *env) {
    float center_x = env->width  * 0.5f;
    float center_y = env->height * 0.5f;
    int   n        = env->num_points;
 
    if (env->method == -1) env->method = rand() % 3;
 
    if (env->method == 0) {
        int opt1 = rand() % n, opt2, opt3, opt4;
        do { opt2 = rand() % n; }
        while (opt2==opt1 || abs(opt2-opt1)==1 || abs(opt2-opt1)==n-1);
        do { opt3 = rand() % n; } while (opt3==opt1 || opt3==opt2);
        do { opt4 = rand() % n; }
        while (opt4==opt1||opt4==opt2||opt4==opt3||
               abs(opt4-opt3)==1||abs(opt4-opt3)==n-1);
 
        for (int i = 0; i < n; i++) {
            float angle = (PI2 * i) / n;
            float dist;
            if      (i==opt1)           dist = env->height*0.2f + (rand()%30);
            else if (i==opt2||i==opt3)  dist = env->height*0.3f + (rand()%40);
            else                        dist = env->height*0.5f + (rand()%30);
            env->track.controls[i].position.x = center_x + dist * cosf(angle);
            env->track.controls[i].position.y = center_y + dist * 0.8f * sinf(angle);
        }
    } else if (env->method == 1) {
        int corner_types[MAX_CONTROL_POINTS], assigned[MAX_CONTROL_POINTS];
        for (int i=0;i<n;i++){corner_types[i]=2;assigned[i]=0;}
 
        int num_med = (n+2)/5;
        for (int placed=0; placed<num_med; placed++) {
            int attempts=0, pos;
            do { pos=rand()%n; if(!assigned[pos]) break; attempts++; } while(attempts<50);
            if (attempts<50){corner_types[pos]=1;assigned[pos]=1;}
        }
        int num_close=(n+2)/8;
        for (int placed=0; placed<num_close; placed++) {
            int attempts=0, pos;
            do {
                pos=rand()%n;
                int pp=(pos-2+n)%n, pr=(pos-1+n)%n, nx=(pos+1)%n;
                bool valid = corner_types[pos]==2 &&
                             corner_types[pp]>0  && corner_types[pr]>0 &&
                             corner_types[nx]>0  && !assigned[pos];
                if (valid) break;
                attempts++;
            } while (attempts<50);
            if (attempts<50){corner_types[pos]=0;assigned[pos]=1;}
        }
        for (int i=0;i<n;i++){
            int pr=(i-1+n)%n, nx=(i+1)%n;
            if(corner_types[pr]==0&&corner_types[i]==2&&corner_types[nx]==0)
                corner_types[i]=1;
        }
        for (int i=0;i<n;i++){
            float angle=(PI2*i)/n, dist;
            if      (corner_types[i]==0) dist=env->height*0.35f+(rand()%30);
            else if (corner_types[i]==1) dist=env->height*0.45f+(rand()%40);
            else                         dist=env->height*0.6f +(rand()%30);
            env->track.controls[i].position.x = center_x + dist*1.2f*cosf(angle);
            env->track.controls[i].position.y = center_y + dist*0.7f*sinf(angle);
        }
    } else {
        float base = env->height*0.5f, var=0.5f, sx=1.0f, sy=0.6f;
        float f1=2.0f+(rand()%5), a1=(1.0f/f1)*(0.9f+0.2f*(rand()%100)/100.0f),
              ph1=PI2*(rand()%100)/100.0f;
        float f2=1.0f+(rand()%2),  a2=0.2f+0.2f*(rand()%100)/100.0f,
              ph2=PI2*(rand()%100)/100.0f;
        float f3=10.0f+0.5f*(rand()%3),a3=0.3f+0.1f*(rand()%100)/100.0f,
              ph3=PI2*(rand()%100)/100.0f;
        for (int i=0;i<n;i++){
            float angle=(PI2*i)/n;
            float rv = a1*cosf(f1*angle+ph1)+a2*cosf(f2*angle+ph2)+a3*cosf(f3*angle+ph3);
            float r  = base + base*var*rv;
            env->track.controls[i].position.x = center_x + r*sx*cosf(angle);
            env->track.controls[i].position.y = center_y + r*sy*sinf(angle);
        }
    }
 
    float tw2 = env->track_width * 0.5f;
    for (int i=0;i<n;i++){
        Vector2 *p = &env->track.controls[i].position;
        if (p->x < tw2)             p->x = tw2;
        if (p->x > env->width-tw2)  p->x = env->width-tw2;
        if (p->y < tw2)             p->y = tw2;
        if (p->y > env->height-tw2) p->y = env->height-tw2;
 
        Vector2 prev = env->track.controls[(i-1+n)%n].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i+1)%n].position;
        float vx1=prev.x-curr.x, vy1=prev.y-curr.y;
        float vx2=next.x-curr.x, vy2=next.y-curr.y;
        float dot=vx1*vx2+vy1*vy2;
        float m1=sqrtf(vx1*vx1+vy1*vy1), m2=sqrtf(vx2*vx2+vy2*vy2);
        if (m1<1e-3f||m2<1e-3f) continue;
        float angle_cos=dot/(m1*m2);
        if (angle_cos > env->corner_thresh) {
            float dx=curr.x-env->width*0.5f, dy=curr.y-env->height*0.5f;
            float dist=sqrtf(dx*dx+dy*dy);
            if (dist>200) {
                float s=0.3f*angle_cos;
                p->x -= dx*s;
                p->y -= dy*s;
            }
        }
    }
}
 
static void GenerateTrackCenterline(WhiskerRacer *env) {
    int point_index = 0;
    for (int i=0; i<env->num_points; i++) {
        Vector2 p0   = env->track.controls[i].position;
        Vector2 p3   = env->track.controls[(i+1)%env->num_points].position;
        Vector2 prev = env->track.controls[(i-1+env->num_points)%env->num_points].position;
        Vector2 next = env->track.controls[(i+2)%env->num_points].position;
 
        Vector2 dir1 = NormalizeVector((Vector2){p3.x-prev.x, p3.y-prev.y});
        Vector2 dir2 = NormalizeVector((Vector2){next.x-p0.x, next.y-p0.y});
 
        float dist = sqrtf((p3.x-p0.x)*(p3.x-p0.x)+(p3.y-p0.y)*(p3.y-p0.y));
        float cl = (i==1||i==3) ? dist*0.2f : (i==0||i==4) ? dist*0.3f : dist*0.4f;
 
        Vector2 p1={(p0.x+dir1.x*cl),(p0.y+dir1.y*cl)};
        Vector2 p2={(p3.x-dir2.x*cl),(p3.y-dir2.y*cl)};
 
        for (int j=0; j<env->bezier_resolution &&
             point_index<MAX_CONTROL_POINTS*env->bezier_resolution-1; j++) {
            float t=(float)j*env->inv_bezier_res;
            env->track.centerline[point_index++]=EvaluateCubicBezier(p0,p1,p2,p3,t);
        }
    }
    env->track.total_points = point_index;
}
 
static void GenerateTrackEdges(WhiskerRacer *env) {
    for (int i=0; i<env->track.total_points; i++) {
        Vector2 cur  = env->track.centerline[i];
        Vector2 nxt  = env->track.centerline[(i+1)%env->track.total_points];
        Vector2 tang = NormalizeVector((Vector2){nxt.x-cur.x, nxt.y-cur.y});
        Vector2 norm = GetPerpendicular(tang);
        float hw = env->track_width * 0.5f;
        env->track.inner_edge[i] = (Vector2){cur.x - norm.x*hw, cur.y - norm.y*hw};
        env->track.outer_edge[i] = (Vector2){cur.x + norm.x*hw, cur.y + norm.y*hw};
    }
}
 
static void GenerateCurbs(WhiskerRacer *env) {
    env->track.curb_count = 0;
    int n = env->num_points;
    for (int i=0; i<n; i++) {
        Vector2 prev = env->track.controls[(i-1+n)%n].position;
        Vector2 curr = env->track.controls[i].position;
        Vector2 next = env->track.controls[(i+1)%n].position;
        float vx1=prev.x-curr.x, vy1=prev.y-curr.y;
        float vx2=next.x-curr.x, vy2=next.y-curr.y;
        float cross=vx1*vy2-vy1*vx2;
        float dot  =vx1*vx2+vy1*vy2;
        float m1=sqrtf(vx1*vx1+vy1*vy1), m2=sqrtf(vx2*vx2+vy2*vy2);
        if (m1<1e-3f||m2<1e-3f) continue;
        float angle_cos=dot/(m1*m2);
        if (angle_cos > -0.8f) {
            int apex_idx = i*env->bezier_resolution;
            Vector2 *edge = (cross>0) ? env->track.inner_edge : env->track.outer_edge;
            for (int j=0;j<4;j++){
                int idx=(apex_idx-1+j+env->track.total_points)%env->track.total_points;
                env->track.curbs[env->track.curb_count][j]=edge[idx];
            }
            env->track.curb_count++;
        }
    }
}
 
static void GenerateRandomTrack(WhiskerRacer *env) {
    GenerateRandomControlPoints(env);
    GenerateTrackCenterline(env);
    GenerateTrackEdges(env);
    GenerateCurbs(env);
}
 
/* =========================================================================
 * Rendering
 * ====================================================================== */
static void TopDownTexture(WhiskerRacer *env,
                           RenderTexture2D *mode7RenderTexture,
                           Vector2 *center_points) {
    if (!env->texture_initialized) {
        *mode7RenderTexture = LoadRenderTexture(env->width, env->height);
        BeginTextureMode(*mode7RenderTexture);
        ClearBackground(DARKGREEN);
        DrawSplineBasis(center_points, env->track.total_points+3,
                        (float)env->track_width, BLACK);
        for (int i=0;i<env->track.curb_count;i++){
            Vector2 cp[4];
            for (int j=0;j<4;j++){
                cp[j]=env->track.curbs[i][j];
                cp[j].y=env->height-cp[j].y;
            }
            DrawSplineBasis(cp,4,5.0f,RED);
        }
        EndTextureMode();
        env->texture_initialized=1;
    }
}
 
static void Mode7(WhiskerRacer *env, RenderTexture2D mode7RenderTexture) {
    BeginDrawing();
    ClearBackground(SKYBLUE);
 
    Texture2D scene = mode7RenderTexture.texture;
    float camX=env->px, camY=env->py, camAngle=env->ang;
    int   height=env->height, width=env->width;
    float inv_w=env->inv_width, inv_h=env->inv_height;
    int   horizon=height/2;
 
    float cos_ang=cosf(camAngle), sin_ang=sinf(camAngle);
    float cos_ang2=-2.0f*cos_ang, sin_ang2=2.0f*sin_ang;
    float height9=9.0f*height;
 
    Image  sceneImage = LoadImageFromTexture(scene);
    Color *pixels     = LoadImageColors(sceneImage);
 
    DrawRectangle(0,horizon,width,height-horizon,DARKGREEN);
    for (int sY=horizon; sY<height; sY+=3) {
        float row=(float)(sY-horizon);
        if(row<1e-4f) row=1e-4f;
        float z=height9/row;
        float dx=-sin_ang*z, dy=cos_ang*z;
        float sx=camX+dy+dx, sy=camY-dx+dy;
        dx=(sin_ang2*z)*inv_w*3.0f;
        dy=(cos_ang2*z)*inv_w*3.0f;
        for (int sX=0; sX<width; sX+=3) {
            int srcX=(int)sx, srcY=(int)sy;
            if(srcX>=0&&srcX<width&&srcY>=0&&srcY<height)
                DrawRectangle(sX,sY,3,3,pixels[srcY*width+srcX]);
            sx+=dx; sy+=dy;
        }
    }
    UnloadImageColors(pixels);
    UnloadImage(sceneImage);
 
    /* Minimap */
    int mmW=width/3, mmH=height/3, mmX=width-mmW, mmY=0;
    DrawTexturePro(scene,(Rectangle){0,0,(float)width,-(float)height},
                   (Rectangle){(float)mmX,(float)mmY,(float)mmW,(float)mmH},
                   (Vector2){0,0},0,WHITE);
 
    float mpx=mmX+(env->px*inv_w)*mmW;
    float mpy=mmY+((height-env->py)*inv_h)*mmH;
    DrawTexturePro(env->puffer,(Rectangle){0,0,128,128},
                   (Rectangle){mpx,mpy,8,8},(Vector2){4,4},
                   (-env->ang*180.0f/PI)-10.0f,WHITE);
 
    /* Overlay car physics sprite */
    car_draw(&env->car, 1.0f, 0.0f, 0.0f, 0.0f, env->height, true);
 
    EndDrawing();
}
 
static void Draw(WhiskerRacer *env, Vector2 *center_points) {
    BeginDrawing();
    ClearBackground(DARKGREEN);
    DrawSplineBasis(center_points, env->track.total_points+3,
                    (float)env->track_width, BLACK);
    for (int i=0;i<env->track.curb_count;i++){
        Vector2 cp[4];
        for (int j=0;j<4;j++){cp[j]=env->track.curbs[i][j]; cp[j].y=env->height-cp[j].y;}
        DrawSplineBasis(cp,4,5.0f,RED);
    }
 
    /* Draw car physics on top of track */
    car_draw(&env->car, 40.0f, 0.0f, 0.0f, 0.0f, env->height, true);
 
    /* Fallback puffer sprite centred on hull */
    float px=env->px, py=env->height-env->py;
    DrawTexturePro(env->puffer,(Rectangle){0,0,128,128},
                   (Rectangle){px,py,48,48},(Vector2){24,24},
                   (-env->ang*180.0f/PI)-10.0f,(Color){255,255,255,255});
 
    EndDrawing();
}
 
void c_render(WhiskerRacer *env) {
    static RenderTexture2D mode7RenderTexture;
 
    env->render=1;
    if (!env->client) env->client=make_client(env);
 
    if (IsKeyDown(KEY_ESCAPE))   exit(0);
    if (IsKeyPressed(KEY_TAB))   ToggleFullscreen();
    if (IsKeyDown(KEY_M))        env->mode7 ^= 1;
 
    if (env->render_many) {
        env->method=rand()%3;
        GenerateRandomTrack(env);
    }
 
    Vector2 *cp=malloc(sizeof(Vector2)*(env->track.total_points+3));
    for (int i=0;i<env->track.total_points;i++){
        cp[i]=env->track.centerline[i];
        cp[i].y=env->height-cp[i].y;
    }
    cp[env->track.total_points  ]=cp[0];
    cp[env->track.total_points+1]=cp[1];
    cp[env->track.total_points+2]=cp[2];
 
    if (env->mode7) {
        TopDownTexture(env,&mode7RenderTexture,cp);
        Mode7(env,mode7RenderTexture);
    } else {
        Draw(env,cp);
    }
    free(cp);
}
 
/* =========================================================================
 * Initialisation / allocation
 * ====================================================================== */
static void init(WhiskerRacer *env) {
    env->tick=0;
    env->debug=0;
    env->inv_width      = 1.0f / env->width;
    env->inv_height     = 1.0f / env->height;
    env->inv_maxv       = 1.0f / env->maxv;
    env->inv_pi2        = 1.0f / PI2;
    env->inv_bezier_res = 1.0f / env->bezier_resolution;
    env->flw_ang        = -env->w_ang;
    env->frw_ang        =  env->w_ang;
    env->texture_initialized=0;
 
    srand(env->rng + env->i);
    GenerateRandomTrack(env);
}
 
void allocate(WhiskerRacer *env) {
    init(env);
    env->observations = (float *)calloc(3, sizeof(float));
    env->actions      = (float *)calloc(1, sizeof(float));
    env->rewards      = (float *)calloc(1, sizeof(float));
    env->terminals    = (float *)calloc(1, sizeof(float));
}
 
/* =========================================================================
 * Step
 * ====================================================================== */
static void step_frame(WhiskerRacer *env, float action) {
    /* --- Discrete/continuous steering --- */
    if (!env->continuous) {
        if (action == LEFT)       env->ang += PI / env->turn_pi_frac;
        else if (action == RIGHT) env->ang -= PI / env->turn_pi_frac;
        if (env->ang > PI2)  env->ang -= PI2;
        else if (env->ang < 0) env->ang += PI2;
 
        /* Map action to car controls */
        car_gas  (&env->car, 1.0f);          /* always throttle in discrete mode */
        car_steer(&env->car, (action==LEFT) ? -1.0f : (action==RIGHT) ? 1.0f : 0.0f);
        car_brake(&env->car, 0.0f);
    } else {
        /* Continuous: action in [-1, 1]; negative = left, positive = right */
        car_steer(&env->car, clampf(action, -1.0f, 1.0f));
        car_gas  (&env->car, 1.0f);
        car_brake(&env->car, 0.0f);
    }
 
    /* Update whisker directions from current heading */
    env->whisker_dirs[0] = (Vector2){cosf(env->ang+env->flw_ang),
                                     sinf(env->ang+env->flw_ang)};
    env->whisker_dirs[1] = (Vector2){cosf(env->ang+env->frw_ang),
                                     sinf(env->ang+env->frw_ang)};
 
    /* --- Advance car physics (60 Hz → dt ≈ 1/60) --- */
    const float DT = 1.0f / 60.0f;
    car_step(&env->car, DT);
 
    /* --- Sync hull position back to legacy scalars --- */
    sync_car_to_env(env);
 
    /* Clamp to screen bounds */
    if (env->px < 0)          env->px = 0;
    else if (env->px > env->width)  env->px = (float)env->width;
    if (env->py < 0)          env->py = 0;
    else if (env->py > env->height) env->py = (float)env->height;
    sync_env_to_car(env);
 
    calc_whisker_lengths(env);
    update_radial_progress(env);
}
 
void c_step(WhiskerRacer *env) {
    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;
 
    float action = env->actions[0];
    for (int i=0; i<env->frameskip; i++) {
        env->tick++;
        step_frame(env, action);
    }
    compute_observations(env);
}