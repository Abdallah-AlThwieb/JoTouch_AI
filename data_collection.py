"""
JoTouch - Data Collection Script
=================================
هذا السكريبت مسؤول عن جمع بيانات التدريب الحقيقية عبر:
  1. قراءة إشارات FMG من حساسات FSR 402 عبر المنفذ التسلسلي (Teensy/Arduino)
  2. استخدام الكاميرا مع MediaPipe لاستخراج زوايا الأصابع تلقائياً (y_angle)
  3. تحديد مرحلة الحركة (Phase) يدوياً عبر لوحة المفاتيح
  4. حفظ البيانات بصيغة .npz جاهزة للتدريب

متطلبات التشغيل:
  pip install mediapipe opencv-python pyserial numpy

الأجهزة المطلوبة:
  - Teensy 4.0 أو أي متحكم يقرأ 8 حساسات FSR ويرسلها عبر Serial
  - كاميرا ويب
"""

import os
import time
import json
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# pyserial مطلوب فقط إذا كان الجهاز متصلاً
try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# =========================
# CONFIG - عدّل هذه القيم حسب إعدادك
# =========================
SERIAL_PORT = "COM3"  # المنفذ التسلسلي للـ Teensy/Arduino (غيّره حسب جهازك)
BAUD_RATE = 115200  # يجب أن يتطابق مع كود الـ Teensy
NUM_SENSORS = 8  # عدد حساسات FSR
TIME_STEPS = 20  # حجم النافذة الزمنية (نفس قيمة النموذج)
SAMPLE_RATE = 100  # Hz - معدل أخذ العينات (100 عينة/ثانية = 10ms لكل عينة)
NUM_DOF = 16  # عدد زوايا المفاصل

# تعريف مراحل الحركة - عدّلها حسب تصميمك
PHASES = {
    0: "Rest      (يد مفتوحة ساكنة)",
    1: "Opening   (فتح اليد)",
    2: "Grasping  (قبضة كاملة)",
}

# مؤشرات مفاصل MediaPipe لحساب 16 زاوية
# MediaPipe يعطي 21 نقطة - نحسب الزوايا من العلاقات بين النقاط
MEDIAPIPE_JOINT_PAIRS = [
    # الإبهام (3 زوايا)
    (1, 2, 3),
    (2, 3, 4),
    (0, 1, 2),
    # السبابة (3 زوايا)
    (5, 6, 7),
    (6, 7, 8),
    (0, 5, 6),
    # الوسطى (3 زوايا)
    (9, 10, 11),
    (10, 11, 12),
    (0, 9, 10),
    # البنصر (3 زوايا)
    (13, 14, 15),
    (14, 15, 16),
    (0, 13, 14),
    # الخنصر (3 زوايا)
    (17, 18, 19),
    (18, 19, 20),
    (0, 17, 18),
    # المعصم العام (1 زاوية)
    (5, 0, 17),
]  # المجموع: 16 زاوية


# =========================
# حساب زاوية المفصل من 3 نقاط
# =========================
def calc_angle(p1, p2, p3) -> float:
    """
    يحسب الزاوية عند النقطة p2 بين الخطين p1-p2 و p2-p3.
    المدخلات: نقاط ثلاثية الأبعاد من MediaPipe
    المخرج: الزاوية بالراديان
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def extract_joint_angles(hand_landmarks) -> np.ndarray:
    """
    يستخرج 16 زاوية مفصل من نقاط MediaPipe Hand Landmarks.
    المدخل: hand_landmarks من MediaPipe
    المخرج: مصفوفة [NUM_DOF] بالراديان
    """
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    angles = []
    for i, j, k in MEDIAPIPE_JOINT_PAIRS:
        angles.append(calc_angle(pts[i], pts[j], pts[k]))
    return np.array(angles, dtype=np.float32)


# =========================
# قراءة إشارات FMG من المنفذ التسلسلي
# =========================
class FMGReader:
    """
    يقرأ بيانات حساسات FSR من الـ Teensy/Arduino عبر Serial.

    تنسيق البيانات المتوقع من الـ Teensy (كل سطر):
      s0,s1,s2,s3,s4,s5,s6,s7\\n
    حيث كل قيمة بين 0 و 1023 (ADC 10-bit) أو 0 و 4095 (ADC 12-bit).

    مثال كود Teensy (Arduino):
      void loop() {
        for (int i = 0; i < 8; i++) {
          Serial.print(analogRead(i));
          if (i < 7) Serial.print(",");
        }
        Serial.println();
        delay(10);  // 100 Hz
      }
    """

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.ser = None
        self.latest_reading = np.zeros(NUM_SENSORS, dtype=np.float32)
        self._running = False
        self._thread = None

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # انتظار تهيئة الجهاز
            print(f"Connected to {self.port} at {self.baud} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False

    def _read_loop(self):
        """خيط مستقل يقرأ البيانات باستمرار."""
        adc_max = 4095.0  # 12-bit لـ Teensy (غيّر لـ 1023.0 إذا استخدمت Arduino)
        while self._running:
            try:
                line = self.ser.readline().decode("utf-8").strip()
                values = [float(v) / adc_max for v in line.split(",")]
                if len(values) == NUM_SENSORS:
                    self.latest_reading = np.array(values, dtype=np.float32)
            except Exception:
                pass

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self.ser:
            self.ser.close()

    def get_reading(self) -> np.ndarray:
        return self.latest_reading.copy()

    @staticmethod
    def list_ports():
        if not SERIAL_AVAILABLE:
            print("pyserial not installed.")
            return
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
        for p in ports:
            print(f"  {p.device} - {p.description}")


# =========================
# محاكي FMG (للاستخدام بدون أجهزة)
# =========================
class FMGSimulator:
    """
    يحاكي قراءات FMG بدون أجهزة حقيقية.
    مفيد لاختبار سكريبت جمع البيانات قبل توصيل الحساسات.
    """

    def __init__(self, phase: int = 0):
        self.phase = phase

    def get_reading(self) -> np.ndarray:
        if self.phase == 0:  # Rest
            base = np.random.uniform(0.05, 0.15, NUM_SENSORS)
        elif self.phase == 1:  # Opening
            base = np.random.uniform(0.3, 0.5, NUM_SENSORS)
        else:  # Grasping
            base = np.random.uniform(0.6, 0.9, NUM_SENSORS)
            base[:4] = np.random.uniform(0.8, 1.0, 4)  # عضلات الانثناء

        return (
            (base + np.random.normal(0, 0.02, NUM_SENSORS))
            .clip(0, 1)
            .astype(np.float32)
        )


# =========================
# جلسة جمع البيانات
# =========================
class DataCollectionSession:
    """
    يدير جلسة جمع البيانات الكاملة.

    التحكم عبر لوحة المفاتيح:
      0, 1, 2    → تحديد المرحلة الحالية (Phase)
      R          → بدء/إيقاف التسجيل للمرحلة المحددة
      S          → حفظ البيانات المجمعة
      Q          → الخروج
    """

    def __init__(self, use_camera: bool = True, use_serial: bool = False):
        self.use_camera = use_camera
        self.use_serial = use_serial
        self.current_phase = 0
        self.recording = False

        # بيانات مجمعة
        self.X_buffer = []  # نوافذ FMG: [N, TIME_STEPS, NUM_SENSORS]
        self.phase_buffer = []  # [N]
        self.angle_buffer = []  # [N, NUM_DOF]

        # نافذة FMG المتراكمة
        self._fmg_window = []

        # إعداد MediaPipe
        if use_camera:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self.mp_draw = mp.solutions.drawing_utils

        # إعداد مصدر FMG
        if use_serial and SERIAL_AVAILABLE:
            self.fmg_source = FMGReader(SERIAL_PORT, BAUD_RATE)
            if not self.fmg_source.connect():
                print("Serial failed. Switching to simulator.")
                self.fmg_source = FMGSimulator(self.current_phase)
                self.use_serial = False
            else:
                self.fmg_source.start()
        else:
            print("Using FMG Simulator (no real hardware).")
            self.fmg_source = FMGSimulator(self.current_phase)

        self.latest_angles = np.zeros(NUM_DOF, dtype=np.float32)

    def _update_fmg_window(self) -> Optional[np.ndarray]:
        """يضيف قراءة جديدة للنافذة ويعيدها عند اكتمالها."""
        reading = self.fmg_source.get_reading()

        # تحديث phase للمحاكي
        if isinstance(self.fmg_source, FMGSimulator):
            self.fmg_source.phase = self.current_phase

        self._fmg_window.append(reading)
        if len(self._fmg_window) >= TIME_STEPS:
            window = np.array(self._fmg_window[-TIME_STEPS:], dtype=np.float32)
            self._fmg_window = self._fmg_window[
                -(TIME_STEPS // 2) :
            ]  # نافذة متداخلة 50%
            return window
        return None

    def _process_camera_frame(self, frame: np.ndarray):
        """يعالج إطار الكاميرا ويستخرج زوايا اليد."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            self.latest_angles = extract_joint_angles(landmarks)
            return True
        return False

    def _draw_ui(self, frame: np.ndarray, hand_detected: bool):
        """يرسم واجهة المستخدم على الإطار."""
        h, w = frame.shape[:2]

        # خلفية شفافة للمعلومات
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # المرحلة الحالية
        phase_text = PHASES.get(self.current_phase, "Unknown")
        color = [(100, 200, 100), (100, 150, 255), (255, 100, 100)][self.current_phase]
        cv2.putText(
            frame,
            f"Phase [{self.current_phase}]: {phase_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # حالة التسجيل
        rec_color = (0, 0, 255) if self.recording else (120, 120, 120)
        rec_text = "● REC" if self.recording else "○ IDLE"
        cv2.putText(
            frame, rec_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2
        )

        # عدد العينات المجمعة
        cv2.putText(
            frame,
            f"Samples: {len(self.X_buffer)}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # حالة اليد
        hand_status = "Hand: Detected" if hand_detected else "Hand: NOT detected"
        hand_color = (0, 255, 100) if hand_detected else (0, 100, 255)
        cv2.putText(
            frame, hand_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1
        )

        # تعليمات
        instructions = "Keys: [0/1/2]=Phase  [R]=Record  [S]=Save  [Q]=Quit"
        cv2.putText(
            frame,
            instructions,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

    def run(self):
        """الحلقة الرئيسية لجمع البيانات."""
        cap = None
        if self.use_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Camera not available. Running without camera.")
                self.use_camera = False
                cap = None

        print("\n" + "=" * 55)
        print("  JoTouch Data Collection")
        print("=" * 55)
        for k, v in PHASES.items():
            print(f"  [{k}] {v}")
        print("\n  [R] Toggle Recording | [S] Save | [Q] Quit")
        if not (cap and self.use_camera):
            print("  (No camera: type commands then press Enter)")
        print("=" * 55 + "\n")

        # خيط مدخل نصي لوضع بدون كاميرا
        if not (cap and self.use_camera):
            self._start_terminal_input_thread()
            print(
                "Ready. Type 0/1/2 to set phase, r to record, s to save, q to quit.\n"
            )

        last_sample_time = time.time()
        sample_interval = 1.0 / SAMPLE_RATE

        while True:
            hand_detected = False

            # --- معالجة الكاميرا ---
            if cap and self.use_camera:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    hand_detected = self._process_camera_frame(frame)
                    self._draw_ui(frame, hand_detected)
                    cv2.imshow("JoTouch Data Collection", frame)
            else:
                # بدون كاميرا: واجهة نصية فقط
                hand_detected = True  # نفترض وجود اليد

            # --- جمع عينة FMG ---
            now = time.time()
            if now - last_sample_time >= sample_interval:
                last_sample_time = now
                window = self._update_fmg_window()

                if window is not None and self.recording:
                    # حفظ العينة فقط إذا تم الكشف عن اليد (أو بدون كاميرا)
                    if hand_detected or not self.use_camera:
                        self.X_buffer.append(window)
                        self.phase_buffer.append(self.current_phase)
                        self.angle_buffer.append(self.latest_angles.copy())

                        if len(self.X_buffer) % 50 == 0:
                            print(
                                f"  Collected {len(self.X_buffer)} samples "
                                f"(Phase {self.current_phase})"
                            )

            # --- معالجة المدخلات ---
            if cap and self.use_camera:
                key = cv2.waitKey(1) & 0xFF
            else:
                # وضع بدون كاميرا: قراءة من خيط الطرفية
                cmd = getattr(self, "_terminal_cmd", None)
                if cmd:
                    self._terminal_cmd = None
                    key = ord(cmd[0]) if cmd else -1
                else:
                    key = -1
                time.sleep(0.01)

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("0"):
                self.current_phase = 0
                print(f"Phase → {PHASES[0]}")
            elif key == ord("1"):
                self.current_phase = 1
                print(f"Phase → {PHASES[1]}")
            elif key == ord("2"):
                self.current_phase = 2
                print(f"Phase → {PHASES[2]}")
            elif key == ord("r") or key == ord("R"):
                self.recording = not self.recording
                status = "STARTED" if self.recording else "STOPPED"
                print(f"Recording {status} | Phase: {PHASES[self.current_phase]}")
            elif key == ord("s") or key == ord("S"):
                self.save_data()

        # تنظيف
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, "_stop_input"):
            self._stop_input = True
        if self.use_serial and hasattr(self.fmg_source, "stop"):
            self.fmg_source.stop()

        # حفظ تلقائي عند الخروج
        if self.X_buffer:
            print("\nAuto-saving before exit...")
            self.save_data()

    def _get_key_no_camera(self) -> int:
        """
        قراءة مدخل لوحة المفاتيح بدون نافذة OpenCV.
        يستخدم خيط مستقل حتى لا يوقف جمع البيانات أثناء انتظار المدخل.
        """
        time.sleep(0.01)
        return -1

    def _start_terminal_input_thread(self):
        """(بدون كاميرا) يفتح خيط مستقل يقرأ أوامر المستخدم من الطرفية."""
        self._terminal_cmd = None
        self._stop_input = False

        def _reader():
            while not self._stop_input:
                try:
                    cmd = input()
                    if cmd:
                        self._terminal_cmd = cmd.strip().lower()
                except EOFError:
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def save_data(self):
        """حفظ البيانات المجمعة إلى ملف .npz"""
        if not self.X_buffer:
            print("No data to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fmg_real_data_{timestamp}.npz"

        X = np.array(self.X_buffer, dtype=np.float32)
        y_phase = np.array(self.phase_buffer, dtype=np.int64)
        y_angle = np.array(self.angle_buffer, dtype=np.float32)

        np.savez(filename, X=X, y_phase=y_phase, y_angle=y_angle)

        # إحصائيات
        print(f"\nSaved: {filename}")
        print(f"  Total samples : {len(X)}")
        print(f"  X shape       : {X.shape}")
        print(f"  y_angle shape : {y_angle.shape}")
        for phase_id, phase_name in PHASES.items():
            count = int((y_phase == phase_id).sum())
            print(f"  Phase [{phase_id}] {phase_name}: {count} samples")

        # حفظ metadata
        meta = {
            "timestamp": timestamp,
            "num_samples": len(X),
            "phases": PHASES,
            "camera_used": self.use_camera,
            "serial_used": self.use_serial,
        }
        with open(f"fmg_session_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("JoTouch Data Collection Script")
    print("=" * 40)

    # اختيار الوضع
    print("\nAvailable modes:")
    print("  [1] Camera + Simulator  (لا أجهزة - للاختبار)")
    print("  [2] Camera + Serial     (كاميرا + Teensy/Arduino)")
    print("  [3] No Camera + Serial  (Teensy فقط بدون كاميرا)")
    print("  [4] List serial ports   (عرض المنافذ المتاحة)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "4":
        FMGReader.list_ports()
    elif choice == "1":
        session = DataCollectionSession(use_camera=True, use_serial=False)
        session.run()
    elif choice == "2":
        session = DataCollectionSession(use_camera=True, use_serial=True)
        session.run()
    elif choice == "3":
        session = DataCollectionSession(use_camera=False, use_serial=True)
        session.run()
    else:
        print("Invalid choice.")
