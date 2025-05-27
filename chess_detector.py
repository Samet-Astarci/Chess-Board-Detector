import sys
import os
import cv2
import numpy as np
import pandas as pd
from stockfish import Stockfish
import math
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QRadioButton ,QSizePolicy,QDialog, QComboBox,QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage ,QIcon
from PyQt5.QtCore import Qt ,QSize ,QTimer,QThread, pyqtSignal



# --- Global ayarlar ---
board_size = 480
cell_size = board_size // 8
colors = [(240, 217, 181), (181, 136, 99)]
model_path = r"C:\Users\samet\Documents\GitHub\Chess-Board-Detector\best4.pt"
assets_folder = "assets"

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_dynamic_point(bbox, ratio):
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    bottom = np.array([(x1 + x2) / 2, y2])
    return (1 - ratio) * center + ratio * bottom

def estimate_angle_ratio(src_points):
    top_len = np.linalg.norm(np.array(src_points[0]) - np.array(src_points[1]))
    bot_len = np.linalg.norm(np.array(src_points[2]) - np.array(src_points[3]))
    ratio = 1 - min(top_len, bot_len) / max(top_len, bot_len)
    ratio = max(0, min(ratio * 1.5, 0.8))
    return ratio

def assign_pieces_dynamic(detections, M, board_size=480, src_points=None):
    if src_points is not None:
        angle_ratio = estimate_angle_ratio(src_points)
    else:
        angle_ratio = 0.4
    cell_size = board_size // 8
    board = [[' ' for _ in range(8)] for _ in range(8)]
    scores = [[-1 for _ in range(8)] for _ in range(8)]
    all_pieces = []

    for det in detections:
        dyn_pt = get_dynamic_point(det['bbox'], angle_ratio)
        dyn_pt_proj = cv2.perspectiveTransform(np.array([[dyn_pt]], dtype=np.float32), M)[0][0]
        px, py = dyn_pt_proj
        col = int(px // cell_size)
        row = int(py // cell_size)
        if 0 <= row < 8 and 0 <= col < 8:
            if det['conf'] > scores[row][col]:
                board[row][col] = det['label']
                scores[row][col] = det['conf']
            all_pieces.append({
                'row': row, 'col': col, 'label': det['label'], 'conf': det['conf']
            })
    return board, all_pieces

def logical_piece_check(all_pieces):
    piece_limits = {
        "white_king": 1,  "black_king": 1,
        "white_queen": 1, "black_queen": 1,
        "white_rook": 2,  "black_rook": 2,
        "white_bishop": 2,"black_bishop": 2,
        "white_knight": 2,"black_knight": 2,
        "white_pawn": 8,  "black_pawn": 8
    }
    piece_counts = {}
    flagged = []
    for piece in all_pieces:
        label = piece['label']
        if label not in piece_counts:
            piece_counts[label] = []
        piece_counts[label].append(piece)
    warnings = []
    for label, items in piece_counts.items():
        limit = piece_limits.get(label, 8)
        if len(items) > limit:
            sorted_items = sorted(items, key=lambda x: x['conf'], reverse=True)
            flagged_pieces = sorted_items[limit:]
            flagged.extend(flagged_pieces)
            kareler = [f"{chr(c+65)}{8-r}" for r,c in [(p['row'],p['col']) for p in flagged_pieces]]
            warnings.append(
                f"⚠️ Fazla '{label}': {len(items)} adet var (limit: {limit}). "
                f"Ekstra taşlar: {', '.join(kareler)}. Diğer olasılıklara bakılmalı!"
            )
    return flagged, warnings

def load_piece_images(cell_size=60):
    types = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors_ = ["white", "black"]
    piece_images = {}
    for color in colors_:
        for ptype in types:
            filename = f"{color}_{ptype}.png"
            path = os.path.join(assets_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
                piece_images[f"{color}_{ptype}"] = resized
    return piece_images

def square_to_coord(square, cell_size=60):
    col = ord(square[0].lower()) - ord('a')
    row = 8 - int(square[1])
    x = col * cell_size + cell_size // 2
    y = row * cell_size + cell_size // 2
    return (x, y)

def draw_board(board, flagged=None, cell_size=60, move_from=None, move_to=None):
    board_img = np.zeros((cell_size*8, cell_size*8, 3), dtype=np.uint8)
    piece_images = load_piece_images(cell_size=cell_size)
    flagged = flagged or []

    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            x, y = col * cell_size, row * cell_size
            cv2.rectangle(board_img, (x, y), (x + cell_size, y + cell_size), color, -1)
            piece = board[row][col]
            if piece != ' ':
                img = piece_images.get(piece)
                if img is not None and img.shape[2] == 4:
                    alpha = img[:, :, 3] / 255.0
                    for c in range(3):
                        board_img[y:y+cell_size, x:x+cell_size, c] = (
                            alpha * img[:, :, c] + (1 - alpha) * board_img[y:y+cell_size, x:x+cell_size, c]
                        ).astype(np.uint8)
                if any(f['row'] == row and f['col'] == col for f in flagged):
                    cv2.rectangle(board_img, (x, y), (x + cell_size, y + cell_size), (0, 0, 255), 3)

    if move_from and move_to:
        x1, y1 = square_to_coord(move_from, cell_size)
        x2, y2 = square_to_coord(move_to, cell_size)
        offset = int(cell_size * 0.4)
        dx = x2 - x1
        dy = y2 - y1
        norm = math.sqrt(dx**2 + dy**2)
        if norm != 0:
            ux, uy = dx / norm, dy / norm
        else:
            ux, uy = 0, 0
        new_x1 = int(x1 + ux * offset)
        new_y1 = int(y1 + uy * offset)
        #cv2.arrowedLine(board_img, (x1, y1), (x2, y2), (30, 200, 80), 6, cv2.LINE_AA, 0, 0.18)
        cv2.arrowedLine(board_img, (new_x1, new_y1), (x2, y2), (0, 200, 60), 6, cv2.LINE_AA, 0, 0.18)
        #cv2.circle(board_img, (x1, y1), 12, (30,200,80), -1) # from
        #cv2.circle(board_img, (x2, y2), 12, (200,80,30), -1) # to

    return board_img
def detect_pieces_ensemble(image_path, model, repeat=5, iou_thresh=0.05, confidence_thresh=0.6):
    """
    Aynı görselde YOLO'yu repeat kadar çalıştırır, kutuları (label+IOU) bazında gruplayıp
    ortalama bbox ve ortalama confidence ile en iyi sonuçları döndürür.
    Çıktı: [{'label': 'white_rook', 'bbox': (x1,y1,x2,y2), 'conf': 0.93}, ...]
    """
    import numpy as np
    from collections import defaultdict

    all_detections = []

    # --- YOLO tekrarları ---
    for i in range(repeat):
        img = cv2.imread(image_path)
        results = model(image_path)[0]
        for r in results.boxes:
            conf = float(r.conf[0])
            if conf < confidence_thresh:
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            label = model.names[int(r.cls)]
            all_detections.append({'label': label, 'bbox': (x1, y1, x2, y2), 'conf': conf})

    # --- Kutu gruplama ---
    grouped = []  # Her grup: {'label', 'boxes': [tuple], 'confs': [float]}
    for det in all_detections:
        found = False
        for group in grouped:
            # Sadece aynı label'dan olanlar gruplanır
            if det['label'] == group['label']:
                # IOU ile yakınlık kontrolü
                for box in group['boxes']:
                    iou = compute_iou(det['bbox'], box)
                    if iou > iou_thresh:
                        group['boxes'].append(det['bbox'])
                        group['confs'].append(det['conf'])
                        found = True
                        break
                if found:
                    break
        if not found:
            grouped.append({
                'label': det['label'],
                'boxes': [det['bbox']],
                'confs': [det['conf']]
            })

    # --- Ortalama bbox ve confidence hesapla ---
    final_detections = []
    for group in grouped:
        # Her bbox'ı ağırlıklı ortalama ile birleştiriyoruz (confidence ile ağırlık!)
        boxes = np.array(group['boxes'])
        confs = np.array(group['confs'])
        w = confs / confs.sum()

        # x1, y1, x2, y2 için ağırlıklı ortalama
        avg_bbox = tuple((boxes * w[:, None]).sum(axis=0).astype(int))
        avg_conf = confs.mean()
        final_detections.append({
            'label': group['label'],
            'bbox': avg_bbox,
            'conf': avg_conf
        })

    return final_detections


def board_to_fen(cell_board, side='white'):
        # Eğer cell_board'un ilk satırı üstse böyle, alt satırsa: reversed_board = cell_board[::-1]
        label_to_fen = {
            "white_king": "K",  "black_king": "k",
            "white_queen": "Q", "black_queen": "q",
            "white_rook": "R",  "black_rook": "r",
            "white_bishop": "B","black_bishop": "b",
            "white_knight": "N","black_knight": "n",
            "white_pawn": "P",  "black_pawn": "p"
        }
        fen_rows = []
        for row in cell_board:
            fen_row = ''
            empty = 0
            for piece in row:
                if piece == ' ':
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += label_to_fen.get(piece, '1')
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        fen_str = '/'.join(fen_rows)
        # Sıra ve kalan alanlar:
        return f"{fen_str} {'w' if side=='white' else 'b'} - - 0 1"
    

STOCKFISH_PATH = r"C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe"

def get_best_move_from_fen(fen_str, stockfish_path=STOCKFISH_PATH, depth=15):
    try:
        stockfish = Stockfish(stockfish_path, depth=depth)
        stockfish.set_fen_position(fen_str)
        move = stockfish.get_best_move()
        return move
    except Exception as e:
        print("Stockfish hatası:", e)
        return None

def empty_board():
    # 8x8 boş
    return [[' ' for _ in range(8)] for _ in range(8)]

# ====================== PYQT5 UI ======================
class ChessUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satranç Analiz Sistemi")
        self.setWindowIcon(QIcon("assets/black_knight.png"))
        self.setGeometry(200, 100, 1100, 600)
        self.selected_image_path = None
        self.corners = []
        self.image_for_display = None
        self.side_color = "white"  # Default A1
        self.analysis_side = "white"
        self.model = YOLO(model_path)
        self.board_img = None
        self.cell_board = None
        self.video_cap = None         # OpenCV VideoCapture nesnesi
        self.video_timer = None       # Qt Timer
        self.current_video_frame = None  # Son alınan frame


        # -- Sol Panel --
        self.btn_select_img = QPushButton("📸 Fotoğraf Seç")
        self.btn_select_img.clicked.connect(self.select_image)

        self.btn_start_video = QPushButton("🎥 Kameradan Canlı")
        self.btn_start_video.clicked.connect(self.open_camera_select_dialog)

        self.btn_capture_frame = QPushButton("🖼️ Kare Al")
        self.btn_capture_frame.setEnabled(False)
        self.btn_capture_frame.clicked.connect(self.capture_current_frame)

        


        self.image_label = QLabel("Henüz görsel yok.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(420, 420)
        self.image_label.mousePressEvent = self.img_mouse_click

        self.corner_info = QLabel("Köşe seçimi: 0/4")
        self.corner_info.setStyleSheet("color: #555")

        self.btn_white_square = QPushButton()
        self.btn_white_square.setIcon(QIcon("assets/white_pawn.png"))
        self.btn_white_square.setIconSize(QSize(32, 32))
        self.btn_white_square.setCheckable(True)
        self.btn_white_square.setChecked(True)
        self.btn_white_square.clicked.connect(self.white_selected)

        self.btn_black_square = QPushButton()
        self.btn_black_square.setIcon(QIcon("assets/black_pawn.png"))
        self.btn_black_square.setIconSize(QSize(32, 32))
        self.btn_black_square.setCheckable(True)
        self.btn_black_square.setChecked(False)
        self.btn_black_square.clicked.connect(self.black_selected)
        
        corner_select_layout = QHBoxLayout()
        corner_select_layout.addWidget(self.btn_white_square)
        corner_select_layout.addWidget(self.btn_black_square)
        


        # -- Sağ Panel --
        right_panel = QVBoxLayout()
        self.analysis_board_label = QLabel()
        self.analysis_board_label.setAlignment(Qt.AlignCenter)
        self.analysis_board_label.setMinimumSize(420, 420)
        self.analysis_board_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # İlk başta boş tahta çiz
        board_img = draw_board(empty_board(), cell_size=52)
        h, w, ch = board_img.shape
        qimg = QImage(board_img.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.analysis_board_label.width(),
            self.analysis_board_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.analysis_board_label.setPixmap(pix)


        self.move_panel = QHBoxLayout()
        self.move_piece_label = QLabel()   # Hamle yapılacak taş ikonu
        self.move_arrow_label = QLabel("→") # Dilersen özel bir ok PNG ekle!
        self.move_info = QLabel()    # E2 → E4

        self.move_panel.addWidget(self.move_piece_label)
        self.move_panel.addWidget(self.move_arrow_label)
        self.move_panel.addWidget(self.move_info)
        self.move_panel.addStretch()

        self.btn_white_for_move = QPushButton()
        self.btn_white_for_move.setIcon(QIcon("assets/white_pawn.png"))
        self.btn_white_for_move.setIconSize(QSize(48,48))
        self.btn_white_for_move.setCheckable(True)
        self.btn_white_for_move.setChecked(True)
        self.btn_white_for_move.clicked.connect(lambda: self.set_analysis_side('white'))

        self.btn_black_for_move = QPushButton()
        self.btn_black_for_move.setIcon(QIcon("assets/black_pawn.png"))
        self.btn_black_for_move.setIconSize(QSize(48,48))
        self.btn_black_for_move.setCheckable(True)
        self.btn_black_for_move.setChecked(False)
        self.btn_black_for_move.clicked.connect(lambda: self.set_analysis_side('black'))

        self.side_select_layout = QVBoxLayout()
        self.side_select_layout.addWidget(self.btn_white_for_move)
        self.side_select_layout.addWidget(self.btn_black_for_move)
        self.side_select_layout.addStretch()

        self.btn_analyze = QPushButton("Analiz Et")
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.btn_analyze.clicked.connect(self.analyze_board)

        # -- Layoutlar --
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.btn_select_img)
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.btn_start_video)
        left_panel.addWidget(self.btn_capture_frame)
        left_panel.addWidget(self.corner_info)
        left_panel.addWidget(QLabel("Alttakilerin tarafı:"))
        left_panel.addLayout(corner_select_layout)
        left_panel.addStretch()
        
        
        # 1. Analiz butonu (üstte, ortada veya sağda)
        analyze_layout = QHBoxLayout()
        analyze_layout.addStretch()
        analyze_layout.addWidget(self.btn_analyze)
        analyze_layout.addStretch()
        right_panel.addLayout(analyze_layout)

        # 2. Orta blok: Satranç tahtası ve side select butonları
        middle_layout = QHBoxLayout()
        middle_layout.addLayout(self.side_select_layout)
        middle_layout.addSpacing(12)
        middle_layout.addWidget(self.analysis_board_label)
        right_panel.addLayout(middle_layout)

        # 3. Alt: hamle paneli
        right_panel.addSpacing(18)
        right_panel.addLayout(self.move_panel)
        right_panel.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel)
        main_layout.addSpacing(16)
        main_layout.addLayout(right_panel)

        self.setLayout(main_layout)

    # --------- UI EVENTLERİ ---------
    def open_camera_select_dialog(self):
        dialog = CameraSelectDialog(self)
        if dialog.exec_():
            selected_source = dialog.get_selected_source()
            if selected_source is None or (isinstance(selected_source, int) and selected_source < 0) or (isinstance(selected_source, str) and not selected_source):
                self.corner_info.setText("Geçerli kamera kaynağı seçilmedi!")
                return
            self.start_video_stream(selected_source)

    def capture_current_frame(self):
        if self.current_video_frame is not None:
            img_path = "temp_capture.jpg"
            cv2.imwrite(img_path, self.current_video_frame)
            img_rgb = cv2.cvtColor(self.current_video_frame, cv2.COLOR_BGR2RGB)
            self.selected_image_path = img_path
            self.image_for_display = img_rgb.copy()
            self.corners = []
            self.update_display_image()
            self.corner_info.setText("Köşe seçimi: 0/4")
            # Video akışını durdur
            self.stop_video_stream()
            self.btn_capture_frame.setEnabled(False)

    def stop_video_stream(self):
        if self.video_timer is not None:
            self.video_timer.stop()
            self.video_timer = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def start_video_stream(self, source=0):
        if self.video_cap is not None:
            self.stop_video_stream()
        self.video_cap = cv2.VideoCapture(source)
        if not self.video_cap.isOpened():
            self.corner_info.setText("Kamera açılamadı! 😢")
            return
        self.corner_info.setText("Canlı görüntü açık. Kare almak için 🖼️ tuşuna bas.")
        self.btn_capture_frame.setEnabled(True)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start(30)


    def update_video_frame(self):
        if self.video_cap is not None and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                self.current_video_frame = frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(pix)

    def white_selected(self):
        self.btn_white_square.setChecked(True)
        self.btn_black_square.setChecked(False)
        self.side_color = "white"

    def black_selected(self):
        self.btn_black_square.setChecked(True)
        self.btn_white_square.setChecked(False)
        self.side_color = "black"
   
    def select_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Fotoğraf Seç', '', 'Images (*.png *.jpg *.jpeg)')
        if fname:
            self.selected_image_path = fname
            self.corners = []
            img = cv2.imread(fname)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_for_display = img_rgb.copy()
            self.update_display_image()
            self.corner_info.setText("Köşe seçimi (Sol Üst): 0/4")
            self.btn_analyze.setEnabled(False)
            self.analysis_board_label.clear()
            self.move_info.setText("")

    def img_mouse_click(self, event):
        if self.image_for_display is None or len(self.corners) >= 4:
            return

        # QLabel boyutları
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        # Orijinal resim boyutları
        img_h, img_w, _ = self.image_for_display.shape

        # Ölçek hesapla (QLabel'a sığdırılmış resmin boyutu ve orijinal oranı)
        scale = min(label_width / img_w, label_height / img_h)
        shown_w = int(img_w * scale)
        shown_h = int(img_h * scale)

        # QLabel'ın ortalanan görselde soldan/üstten boşluk payı (padding) var
        offset_x = (label_width - shown_w) // 2
        offset_y = (label_height - shown_h) // 2

        # Tıklanan noktanın QLabel içindeki konumu
        click_x = event.pos().x()
        click_y = event.pos().y()

        # Eğer tıklama görsel alanı dışında ise işlem yapma
        if not (offset_x <= click_x <= offset_x + shown_w and offset_y <= click_y <= offset_y + shown_h):
            return

        # Tıklamanın, gösterilen (scaled) resim içindeki yeri
        rel_x = click_x - offset_x
        rel_y = click_y - offset_y

        # Orijinal resim koordinatına çevir
        orig_x = int(rel_x / scale)
        orig_y = int(rel_y / scale)

        self.corners.append((orig_x, orig_y))

        corner_names = ["Sol Üst", "Sağ Üst", "Sağ Alt", "Sol Alt"]
        if len(self.corners) < 4:
            self.corner_info.setText(f"Köşe seçimi ({corner_names[len(self.corners)]}): {len(self.corners)}/4")

        self.update_display_image()
        if len(self.corners) == 4:
            self.corner_info.setText("Köşe seçimi tamam! Analiz edebilirsin.")
            print(str(self.corners)+" tüm noktalar")
            self.btn_analyze.setEnabled(True)


    def update_display_image(self):
        if self.image_for_display is None:
            return
        img = self.image_for_display.copy()
        for i, (x, y) in enumerate(self.corners):
            cv2.circle(img, (x, y), 10, (255,0,0), -1)
            cv2.putText(img, f"{i+1}", (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(420, 420, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    def analyze_board(self):
        if not self.selected_image_path or len(self.corners) != 4:
            self.move_info.setText("Görsel ve 4 köşe seçimi gerekli!")
            return
        src_pts = np.array(self.corners, dtype="float32")
        if self.side_color == "white":  # Sol alt BEYAZ ise A1
            dst_pts = np.array([
                [0, 0],
                [board_size - 1, 0],
                [board_size - 1, board_size - 1],
                [0, board_size - 1]
            ], dtype="float32")
        else:  # Sol alt SİYAH ise H1
            dst_pts = np.array([
                [board_size - 1, 0],
                [0, 0],
                [0, board_size - 1],
                [board_size - 1, board_size - 1]
            ], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        #img = cv2.imread(self.selected_image_path)
        #board_img = cv2.warpPerspective(img, M, (board_size, board_size))

        # *** ÖNEMLİ DEĞİŞİKLİK: Detection orijinal resimde ***
        detections = detect_pieces_ensemble(self.selected_image_path, self.model, repeat=1, confidence_thresh=0.5)
        # *** KUTULARIN PRESPEKTİF DÜZELTİLMESİ ***
        cell_board, all_pieces = assign_pieces_dynamic(detections, M, board_size=board_size, src_points=self.corners)
        self.cell_board = cell_board

        # --- Mantıksal kontrol, uyarılar ---
        flagged, warnings = logical_piece_check(all_pieces)
        if warnings:
            self.move_info.setText('\n'.join(warnings))

        # --- Tahtayı çiz ---
        vis = draw_board(cell_board, flagged=flagged, cell_size=52)
        h, w, ch = vis.shape
        qimg = QImage(vis.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(420, 420, Qt.KeepAspectRatio)
        self.analysis_board_label.setPixmap(pix)

        # --- En iyi hamle ---
        self.show_best_move()
    
    def set_analysis_side(self, side):
        # Toggle buttonların görünümünü güncelle
        if side == 'white':
            self.btn_white_for_move.setChecked(True)
            self.btn_black_for_move.setChecked(False)
        else:
            self.btn_white_for_move.setChecked(False)
            self.btn_black_for_move.setChecked(True)
        self.analysis_side = side
        # Anında öneriyi güncellemek için (eğer cell_board mevcutsa)
        if self.cell_board is not None:
            self.show_best_move()

    def detect_moved_piece_label(self, from_square):
        """
        from_square: string, örn 'E2'
        self.cell_board: 8x8 tahta (row, col)
        Dönüş: 'white_pawn' gibi bir label (asset adı)
        """
        # Satrançta A-H ve 8-1 arası
        # A: col 0, H: col 7 / 8: row 0, 1: row 7
        col = ord(from_square[0].upper()) - ord('A')
        row = 8 - int(from_square[1])
        if 0 <= row < 8 and 0 <= col < 8:
            return self.cell_board[row][col]
        else:
            return None

    def show_best_move(self):
        if self.cell_board is not None:
            fen = board_to_fen(self.cell_board, side=self.analysis_side)
            move = get_best_move_from_fen(fen)
            if move:
                # Örn: 'e2e4' → 'E2', 'E4'
                from_sq = move[:2].upper()
                to_sq = move[2:].upper()
                # Taşı bul ve PNG'sini göster
                moved_piece_label = self.detect_moved_piece_label(from_sq)
                asset_path = f"assets/{moved_piece_label}.png"
                self.move_piece_label.setPixmap(QPixmap(asset_path).scaled(40,40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.move_arrow_label.setText("→")
                self.move_info.setText(f"{from_sq} → {to_sq}")
                from_sq = move[:2]
                to_sq = move[2:]
                vis = draw_board(self.cell_board, cell_size=52, move_from=from_sq, move_to=to_sq)
                # ... (QImage, QPixmap ile analysis_board_label'a bas)
                h, w, ch = vis.shape
                qimg = QImage(vis.data, w, h, ch*w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.analysis_board_label.width(),
                    self.analysis_board_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.analysis_board_label.setPixmap(pix)
            else:
                self.move_info.setText("Hamle bulunamadı!")
                self.move_piece_label.clear()
                self.move_arrow_label.clear()


class CameraScanThread(QThread):
    cameras_found = pyqtSignal(list)
    def __init__(self, max_cams=5):
        super().__init__()
        self.max_cams = max_cams
    def run(self):
        found = []
        for i in range(self.max_cams):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
                cap.release()
        self.cameras_found.emit(found)

class CameraSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera Kaynağı Seç")
        self.setWindowIcon(QIcon("assets/black_rook.png"))
        self.setMinimumWidth(350)
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Kaynak türü seçimi
        self.radio_webcam = QRadioButton("Webcam")
        self.radio_ipcam = QRadioButton("IP Kamera (IP Webcam / Telefon)")
        self.radio_webcam.setChecked(True)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_webcam)
        radio_layout.addWidget(self.radio_ipcam)
        layout.addLayout(radio_layout)

        # Webcam seçimi
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItem("Kameralar aranıyor...", -1)
        self.webcam_combo.setEnabled(False)
        layout.addWidget(QLabel("Webcam Seç:"))
        layout.addWidget(self.webcam_combo)

        # IP Kamera girişi
        self.ipcam_edit = QLineEdit()
        self.ipcam_edit.setPlaceholderText("örn: http://192.168.0.21:8080/video")
        self.ipcam_edit.setText("http://192.168.0.21:8080/video")
        self.ipcam_edit.setEnabled(False)
        layout.addWidget(QLabel("IP Kamera Adresi:"))
        layout.addWidget(self.ipcam_edit)

        # Radio button seçimine göre aktif/pasif alanlar
        self.radio_webcam.toggled.connect(self.update_fields)
        self.radio_ipcam.toggled.connect(self.update_fields)

        # Butonlar
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Tamam")
        self.btn_cancel = QPushButton("İptal")
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # Kamera taramasını thread'de başlat
        self.scan_thread = CameraScanThread(max_cams=5)
        self.scan_thread.cameras_found.connect(self.update_webcam_list)
        self.scan_thread.start()

    def update_fields(self):
        if self.radio_webcam.isChecked():
            self.webcam_combo.setEnabled(True)
            self.ipcam_edit.setEnabled(False)
        else:
            self.webcam_combo.setEnabled(False)
            self.ipcam_edit.setEnabled(True)

    def update_webcam_list(self, cams):
        self.webcam_combo.clear()
        if not cams:
            self.webcam_combo.addItem("Hiç kamera bulunamadı!", -1)
            self.webcam_combo.setEnabled(False)
        else:
            for idx in cams:
                self.webcam_combo.addItem(f"Webcam {idx}", idx)
            self.webcam_combo.setEnabled(True)

    def get_selected_source(self):
        if self.radio_webcam.isChecked():
            cam_index = self.webcam_combo.currentData()
            return cam_index  # int
        else:
            url = self.ipcam_edit.text().strip()
            if url and not url.endswith("/video"):
                if url[-1] != "/":
                    url += "/"
                url += "video"
            return url

# ---- Ana uygulama çalıştırıcı ----
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessUI()
    window.show()
    sys.exit(app.exec_())
