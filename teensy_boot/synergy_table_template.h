// JoTouch Stage-2 synergy table — TEMPLATE (fill from your hand kinematics)
// motor_cmd[m] = intensity * SYNERGY_TABLE[gesture][m]
// intensity comes from the Stage-1 boot model (0.0 .. 1.0)
// Each row = the 7-motor target posture (normalized 0..1 of each motor's
// full travel) for one gesture at full closure.

#ifndef JOTOUCH_SYNERGY_TABLE_H
#define JOTOUCH_SYNERGY_TABLE_H

#define NUM_MOTORS   7
#define NUM_GESTURES 4   // adjust to your gesture set

// Gestures (example order — match your classifier's class order):
//   0 = REST, 1 = OPEN/WAVE, 2 = PINCH, 3 = POWER GRIP
static const float SYNERGY_TABLE[NUM_GESTURES][NUM_MOTORS] = {
  // M1    M2    M3    M4    M5    M6    M7   <- your 7 motors
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // REST
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // OPEN  (fill: full extension)
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // PINCH (fill: thumb+index posture)
  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f },  // GRIP  (example: all finger motors close)
};

// The 12-16 finger DOFs follow mechanically from the 7 motors via the
// tendon/linkage coupling (Stage 3) — no software mapping needed.

#endif  // JOTOUCH_SYNERGY_TABLE_H
