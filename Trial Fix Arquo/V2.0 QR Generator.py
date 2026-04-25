"""
V2.0 QR & ArUco Generator - T-MIND Project
Generates QR codes and ArUco markers with a tabbed GUI.
Saves output to the Export folder.
"""

import qrcode
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from PIL import Image, ImageTk
from datetime import datetime

# ============================================================
# CONFIGURATION  —  Ubah nilai di bawah ini sesuai kebutuhan
# ============================================================

# --- Path export ---
EXPORT_DIR = r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\First Trial (Snitching 3 CV)\Export QR Generator"

# --- Default QR data & filename ---
DEFAULT_QR_DATA     = "Arya Dwi Kusuma"                # Isi default teks QR (kosong = user isi manual)
DEFAULT_FILENAME    = "ExportQR_Arya Dwi Kusuma"       # Nama file default (tanpa .png)

# --- QR parameters ---
DEFAULT_BOX_SIZE    = 10                # Ukuran piksel per modul QR (5-30)
DEFAULT_BORDER      = 4                 # Ketebalan border dalam modul (1-10)
DEFAULT_ERROR_CORR  = "L (7%)"          # Level koreksi: "L (7%)", "M (15%)", "Q (25%)", "H (30%)"

# --- QR Version (kompleksitas grid) ---
# Semakin kecil version = semakin simpel QR
# None = auto-detect berdasarkan panjang data
#   Version 1  = 21×21 modul  (max 17 huruf dgn L)   ← PALING SIMPEL
#   Version 2  = 25×25 modul  (max 32 huruf dgn L)
#   Version 3  = 29×29 modul  (max 53 huruf dgn L)
#   Version 4  = 33×33 modul  (max 78 huruf dgn L)
#   Version 5  = 37×37 modul  (max 106 huruf dgn L)
# Tips: Untuk CV/kamera, pakai version sekecil mungkin + data pendek
DEFAULT_QR_VERSION  = 1                 # 1-40, atau None untuk auto

# --- Output image size ---
# Rumus: Piksel = CM × DPI ÷ 2.54
# Tabel acuan CM → Piksel (pada 300 DPI, kualitas cetak):
#   3 cm  = 354 px    |   5 cm  = 591 px    |   7 cm  = 827 px
#   4 cm  = 472 px    |   6 cm  = 709 px    |   8 cm  = 945 px
#   10 cm = 1181 px   |   15 cm = 1772 px   |   20 cm = 2362 px
# Pada 150 DPI (cetak biasa): bagi 2 dari nilai di atas
# Pada 72 DPI (layar/web):    bagi ~4 dari nilai di atas

DEFAULT_WIDTH       = 1181               # Lebar gambar output dalam piksel (100-4000)
DEFAULT_HEIGHT      = 1181               # Tinggi gambar output dalam piksel (100-4000)

# --- Warna ---
# Contoh warna (hex):
#   Hitam   = "#000000"    |   Putih    = "#ffffff"
#   Merah   = "#ff0000"    |   Hijau    = "#00ff00"    |   Biru     = "#0000ff"
#   Kuning  = "#ffff00"    |   Cyan     = "#00ffff"    |   Magenta  = "#ff00ff"
#   Abu-abu = "#808080"    |   Orange   = "#ff8c00"    |   Ungu     = "#800080"
#   Navy    = "#000080"    |   Teal     = "#008080"    |   Maroon   = "#800000"

DEFAULT_QR_COLOR    = "#000000"         # Warna QR code (hex)
DEFAULT_BG_COLOR    = "#ffffff"         # Warna background (hex)

# --- ArUco Marker defaults ---
# Dictionary types (ukuran grid):
#   "4x4_50"   = 4×4 grid, 50 marker   ← PALING SIMPEL, cocok CV
#   "4x4_100"  = 4×4 grid, 100 marker
#   "4x4_250"  = 4×4 grid, 250 marker
#   "5x5_50"   = 5×5 grid, 50 marker
#   "5x5_100"  = 5×5 grid, 100 marker
#   "5x5_250"  = 5×5 grid, 250 marker
#   "6x6_50"   = 6×6 grid, 50 marker
#   "6x6_250"  = 6×6 grid, 250 marker
#   "7x7_50"   = 7×7 grid, 50 marker
DEFAULT_ARUCO_DICT      = "4x4_50"      # Dictionary type
DEFAULT_ARUCO_ID        = 0             # ID marker (0 sampai max-1)
DEFAULT_ARUCO_SIZE      = 700           # Ukuran output marker dalam piksel
DEFAULT_ARUCO_BORDER    = 2             # Border bits (1-3)
DEFAULT_ARUCO_FILENAME  = "ArUco_ID0"   # Nama file default

# --- GUI window ---
WINDOW_TITLE        = " "
WINDOW_WIDTH        = 620               # Lebar jendela GUI
WINDOW_HEIGHT       = 780               # Tinggi jendela GUI

# --- Preview ---
PREVIEW_SIZE        = 240               # Ukuran preview dalam piksel

# ============================================================

# Ensure export directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "4x4_50":   cv2.aruco.DICT_4X4_50,
    "4x4_100":  cv2.aruco.DICT_4X4_100,
    "4x4_250":  cv2.aruco.DICT_4X4_250,
    "5x5_50":   cv2.aruco.DICT_5X5_50,
    "5x5_100":  cv2.aruco.DICT_5X5_100,
    "5x5_250":  cv2.aruco.DICT_5X5_250,
    "6x6_50":   cv2.aruco.DICT_6X6_50,
    "6x6_250":  cv2.aruco.DICT_6X6_250,
    "7x7_50":   cv2.aruco.DICT_7X7_50,
}


# ============================================================
# QR GENERATION LOGIC
# ============================================================
def generate_qr(data, filename, fill_color=DEFAULT_QR_COLOR, back_color=DEFAULT_BG_COLOR,
                box_size=DEFAULT_BOX_SIZE, border=DEFAULT_BORDER,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                output_width=DEFAULT_WIDTH, output_height=DEFAULT_HEIGHT,
                version=DEFAULT_QR_VERSION):
    """Generate a QR code image and save it."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

    # Resize to custom dimensions if specified
    if output_width and output_height:
        img = img.resize((output_width, output_height), Image.NEAREST)

    # Build full path
    if not filename.lower().endswith(".png"):
        filename += ".png"
    filepath = os.path.join(EXPORT_DIR, filename)
    img.save(filepath)
    return filepath, img


# ============================================================
# ARUCO GENERATION LOGIC
# ============================================================
def generate_aruco(marker_id, dict_type, size, border_bits, filename):
    """Generate an ArUco marker image and save it."""
    dict_key = ARUCO_DICT_MAP.get(dict_type, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)

    # Generate marker
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size, borderBits=border_bits)

    # Convert to PIL Image (grayscale → RGB)
    pil_img = Image.fromarray(marker_img).convert("RGB")

    # Build full path
    if not filename.lower().endswith(".png"):
        filename += ".png"
    filepath = os.path.join(EXPORT_DIR, filename)
    pil_img.save(filepath)
    return filepath, pil_img


# ============================================================
# GUI APPLICATION
# ============================================================
class QRGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.fill_color = DEFAULT_QR_COLOR
        self.back_color = DEFAULT_BG_COLOR
        self.preview_photo = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Dark theme styles
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                        foreground="#cdd6f4", background="#1e1e2e")
        style.configure("Sub.TLabel", font=("Segoe UI", 9),
                        foreground="#a6adc8", background="#1e1e2e")
        style.configure("Field.TLabel", font=("Segoe UI", 10, "bold"),
                        foreground="#bac2de", background="#1e1e2e")
        style.configure("Card.TFrame", background="#313244")
        style.configure("Main.TFrame", background="#1e1e2e")
        style.configure("TCombobox", fieldbackground="#45475a",
                        background="#45475a", foreground="#cdd6f4")

        # Tab styles
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"),
                        background="#45475a", foreground="#cdd6f4",
                        padding=[16, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", "#89b4fa")],
                  foreground=[("selected", "#1e1e2e")])

        # Card style for tabs
        style.configure("Tab.TFrame", background="#1e1e2e")

        # Main container
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill="both", expand=True, padx=16, pady=10)

        # Title
        ttk.Label(main, text="⬡  QR & ArUco Generator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text=f"Export → {EXPORT_DIR}", style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        # ======== TABBED NOTEBOOK ========
        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True, pady=(0, 6))

        # ---------- TAB 1: QR Code ----------
        qr_tab = ttk.Frame(notebook, style="Tab.TFrame")
        notebook.add(qr_tab, text="  📱 QR Code  ")
        self._build_qr_tab(qr_tab)

        # ---------- TAB 2: ArUco Marker ----------
        aruco_tab = ttk.Frame(notebook, style="Tab.TFrame")
        notebook.add(aruco_tab, text="  🎯 ArUco Marker  ")
        self._build_aruco_tab(aruco_tab)

        # ======== SHARED PREVIEW ========
        preview_card = ttk.Frame(main, style="Card.TFrame")
        preview_card.pack(fill="both", expand=True, pady=(6, 0))
        ttk.Label(preview_card, text="Preview", style="Field.TLabel").pack(anchor="w", padx=14, pady=(8, 0))
        self.preview_label = tk.Label(preview_card, bg="#313244",
                                      text="Preview akan muncul di sini",
                                      fg="#6c7086", font=("Segoe UI", 10))
        self.preview_label.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(main, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg="#1e1e2e", fg="#6c7086", anchor="w").pack(fill="x")

    # ================================================================
    # QR TAB
    # ================================================================
    def _build_qr_tab(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=6, pady=6)
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        # Data input
        ttk.Label(inner, text="QR Data / Text", style="Field.TLabel").pack(anchor="w")
        self.data_text = tk.Text(inner, height=3, width=50, font=("Consolas", 10),
                                bg="#45475a", fg="#cdd6f4", insertbackground="#cdd6f4",
                                relief="flat", bd=0, wrap="word")
        self.data_text.pack(fill="x", pady=(3, 8))
        if DEFAULT_QR_DATA:
            self.data_text.insert("1.0", DEFAULT_QR_DATA)

        # Filename
        ttk.Label(inner, text="Filename (tanpa .png)", style="Field.TLabel").pack(anchor="w")
        self.filename_var = tk.StringVar(value=DEFAULT_FILENAME)
        tk.Entry(inner, textvariable=self.filename_var, font=("Consolas", 10),
                 bg="#45475a", fg="#cdd6f4", insertbackground="#cdd6f4",
                 relief="flat", bd=0).pack(fill="x", ipady=4, pady=(3, 8))

        # Options Row
        opts = ttk.Frame(inner, style="Card.TFrame")
        opts.pack(fill="x", pady=(0, 4))

        for col, (label, var_val, fr, to) in enumerate([
            ("Box Size", DEFAULT_BOX_SIZE, 5, 30),
            ("Border", DEFAULT_BORDER, 1, 10),
        ]):
            ttk.Label(opts, text=label, style="Field.TLabel").grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 16, 0))
            var = tk.IntVar(value=var_val)
            tk.Spinbox(opts, from_=fr, to=to, textvariable=var, width=5,
                       font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                       relief="flat", bd=0, buttonbackground="#585b70"
                       ).grid(row=1, column=col, sticky="w", padx=(0 if col == 0 else 16, 0), pady=(2, 0))
            if col == 0: self.box_size_var = var
            else: self.border_var = var

        # Error correction
        ttk.Label(opts, text="Error Corr.", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.ec_var = tk.StringVar(value=DEFAULT_ERROR_CORR)
        ttk.Combobox(opts, textvariable=self.ec_var, width=10, state="readonly",
                     values=["L (7%)", "M (15%)", "Q (25%)", "H (30%)"]
                     ).grid(row=1, column=2, sticky="w", padx=(16, 0), pady=(2, 0))

        # Size row
        sz = ttk.Frame(inner, style="Card.TFrame")
        sz.pack(fill="x", pady=(6, 4))

        for col, (label, var_val) in enumerate([("Width (px)", DEFAULT_WIDTH), ("Height (px)", DEFAULT_HEIGHT)]):
            ttk.Label(sz, text=label, style="Field.TLabel").grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else 16, 0))
            var = tk.IntVar(value=var_val)
            tk.Spinbox(sz, from_=100, to=4000, increment=50, textvariable=var, width=6,
                       font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                       relief="flat", bd=0, buttonbackground="#585b70"
                       ).grid(row=1, column=col, sticky="w", padx=(0 if col == 0 else 16, 0), pady=(2, 0))
            if col == 0: self.width_var = var
            else: self.height_var = var

        # Color pickers
        cf = ttk.Frame(inner, style="Card.TFrame")
        cf.pack(fill="x", pady=(6, 4))

        ttk.Label(cf, text="QR Color", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.fill_btn = tk.Button(cf, bg=self.fill_color, width=4, height=1,
                                  relief="flat", command=lambda: self._pick_color("fill"))
        self.fill_btn.grid(row=0, column=1, padx=(6, 16))

        ttk.Label(cf, text="Background", style="Field.TLabel").grid(row=0, column=2, sticky="w")
        self.back_btn = tk.Button(cf, bg=self.back_color, width=4, height=1,
                                  relief="flat", command=lambda: self._pick_color("back"))
        self.back_btn.grid(row=0, column=3, padx=(6, 0))

        # Generate QR button
        tk.Button(inner, text="⚡  Generate QR Code", font=("Segoe UI", 11, "bold"),
                  bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
                  activeforeground="#1e1e2e", relief="flat", bd=0, cursor="hand2",
                  command=self._on_generate_qr).pack(fill="x", ipady=6, pady=(8, 0))

    # ================================================================
    # ARUCO TAB
    # ================================================================
    def _build_aruco_tab(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=6, pady=6)
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        # Info
        ttk.Label(inner, text="ArUco Marker — simpel, cepat, ideal untuk CV",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        # Row 1: Dictionary + ID
        r1 = ttk.Frame(inner, style="Card.TFrame")
        r1.pack(fill="x", pady=(0, 6))

        ttk.Label(r1, text="Dictionary", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.aruco_dict_var = tk.StringVar(value=DEFAULT_ARUCO_DICT)
        dict_combo = ttk.Combobox(r1, textvariable=self.aruco_dict_var, width=12, state="readonly",
                                  values=list(ARUCO_DICT_MAP.keys()))
        dict_combo.grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Label(r1, text="Marker ID", style="Field.TLabel").grid(row=0, column=1, sticky="w", padx=(20, 0))
        self.aruco_id_var = tk.IntVar(value=DEFAULT_ARUCO_ID)
        tk.Spinbox(r1, from_=0, to=249, textvariable=self.aruco_id_var, width=5,
                   font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                   relief="flat", bd=0, buttonbackground="#585b70"
                   ).grid(row=1, column=1, sticky="w", padx=(20, 0), pady=(2, 0))

        ttk.Label(r1, text="Border Bits", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.aruco_border_var = tk.IntVar(value=DEFAULT_ARUCO_BORDER)
        tk.Spinbox(r1, from_=1, to=3, textvariable=self.aruco_border_var, width=4,
                   font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                   relief="flat", bd=0, buttonbackground="#585b70"
                   ).grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(2, 0))

        # Row 2: Size + Filename
        r2 = ttk.Frame(inner, style="Card.TFrame")
        r2.pack(fill="x", pady=(6, 6))

        ttk.Label(r2, text="Size (px)", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.aruco_size_var = tk.IntVar(value=DEFAULT_ARUCO_SIZE)
        tk.Spinbox(r2, from_=100, to=4000, increment=50, textvariable=self.aruco_size_var, width=6,
                   font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                   relief="flat", bd=0, buttonbackground="#585b70"
                   ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Label(r2, text="Filename (tanpa .png)", style="Field.TLabel").grid(row=0, column=1, sticky="w", padx=(20, 0))
        self.aruco_filename_var = tk.StringVar(value=DEFAULT_ARUCO_FILENAME)
        tk.Entry(r2, textvariable=self.aruco_filename_var, font=("Consolas", 10), width=22,
                 bg="#45475a", fg="#cdd6f4", insertbackground="#cdd6f4",
                 relief="flat", bd=0).grid(row=1, column=1, sticky="w", padx=(20, 0), ipady=4, pady=(2, 0))

        # Batch generate
        r3 = ttk.Frame(inner, style="Card.TFrame")
        r3.pack(fill="x", pady=(6, 6))

        self.aruco_batch_var = tk.BooleanVar(value=False)
        tk.Checkbutton(r3, text="Batch Generate (ID 0 sampai N)",
                       variable=self.aruco_batch_var,
                       font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4",
                       activebackground="#313244", activeforeground="#cdd6f4",
                       selectcolor="#45475a").grid(row=0, column=0, sticky="w")

        ttk.Label(r3, text="Sampai ID:", style="Field.TLabel").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.aruco_batch_max_var = tk.IntVar(value=9)
        tk.Spinbox(r3, from_=1, to=249, textvariable=self.aruco_batch_max_var, width=4,
                   font=("Consolas", 10), bg="#45475a", fg="#cdd6f4",
                   relief="flat", bd=0, buttonbackground="#585b70"
                   ).grid(row=0, column=2, sticky="w", padx=(6, 0))

        # Reference table
        ref = ttk.Label(inner, text=(
            "📋 Dictionary reference:\n"
            "  4x4_50  = 4×4 grid, ID 0-49   ← Paling simpel\n"
            "  5x5_50  = 5×5 grid, ID 0-49\n"
            "  6x6_50  = 6×6 grid, ID 0-49\n"
            "  Pakai 4x4 untuk deteksi tercepat di OpenCV"
        ), style="Sub.TLabel", justify="left")
        ref.pack(anchor="w", pady=(4, 6))

        # Generate ArUco button
        tk.Button(inner, text="🎯  Generate ArUco Marker", font=("Segoe UI", 11, "bold"),
                  bg="#a6e3a1", fg="#1e1e2e", activebackground="#94e2d5",
                  activeforeground="#1e1e2e", relief="flat", bd=0, cursor="hand2",
                  command=self._on_generate_aruco).pack(fill="x", ipady=6, pady=(4, 0))

    # ================================================================
    # HELPERS
    # ================================================================
    def _pick_color(self, which):
        color = colorchooser.askcolor(title=f"Pick {'QR' if which == 'fill' else 'Background'} Color")
        if color[1]:
            if which == "fill":
                self.fill_color = color[1]
                self.fill_btn.configure(bg=self.fill_color)
            else:
                self.back_color = color[1]
                self.back_btn.configure(bg=self.back_color)

    def _get_error_correction(self):
        mapping = {
            "L (7%)":  qrcode.constants.ERROR_CORRECT_L,
            "M (15%)": qrcode.constants.ERROR_CORRECT_M,
            "Q (25%)": qrcode.constants.ERROR_CORRECT_Q,
            "H (30%)": qrcode.constants.ERROR_CORRECT_H,
        }
        return mapping.get(self.ec_var.get(), qrcode.constants.ERROR_CORRECT_H)

    def _show_preview(self, img):
        """Show a PIL Image in the preview area."""
        img_resized = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(img_resized)
        self.preview_label.configure(image=self.preview_photo, text="")

    # ================================================================
    # QR GENERATE ACTION
    # ================================================================
    def _on_generate_qr(self):
        data = self.data_text.get("1.0", "end").strip()
        filename = self.filename_var.get().strip()

        if not data:
            messagebox.showwarning("Input Kosong", "Masukkan data/teks untuk QR code.")
            return
        if not filename:
            messagebox.showwarning("Filename Kosong", "Masukkan nama file.")
            return

        try:
            filepath, img = generate_qr(
                data=data,
                filename=filename,
                fill_color=self.fill_color,
                back_color=self.back_color,
                box_size=self.box_size_var.get(),
                border=self.border_var.get(),
                error_correction=self._get_error_correction(),
                output_width=self.width_var.get(),
                output_height=self.height_var.get(),
            )

            self._show_preview(img)
            self.status_var.set(f"✅  QR Saved → {filepath}")
            messagebox.showinfo("Berhasil", f"QR Code disimpan di:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal generate QR:\n{e}")
            self.status_var.set(f"❌ Error: {e}")

    # ================================================================
    # ARUCO GENERATE ACTION
    # ================================================================
    def _on_generate_aruco(self):
        dict_type = self.aruco_dict_var.get()
        size = self.aruco_size_var.get()
        border_bits = self.aruco_border_var.get()

        try:
            if self.aruco_batch_var.get():
                # Batch mode
                max_id = self.aruco_batch_max_var.get()
                count = 0
                last_img = None
                for mid in range(0, max_id + 1):
                    fname = f"ArUco_{dict_type}_ID{mid}"
                    filepath, img = generate_aruco(mid, dict_type, size, border_bits, fname)
                    last_img = img
                    count += 1

                if last_img:
                    self._show_preview(last_img)
                self.status_var.set(f"✅  Batch: {count} ArUco markers saved")
                messagebox.showinfo("Berhasil",
                    f"{count} ArUco markers (ID 0-{max_id}) disimpan di:\n{EXPORT_DIR}")
            else:
                # Single mode
                marker_id = self.aruco_id_var.get()
                filename = self.aruco_filename_var.get().strip()
                if not filename:
                    filename = f"ArUco_{dict_type}_ID{marker_id}"

                filepath, img = generate_aruco(marker_id, dict_type, size, border_bits, filename)
                self._show_preview(img)
                self.status_var.set(f"✅  ArUco Saved → {filepath}")
                messagebox.showinfo("Berhasil", f"ArUco Marker disimpan di:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal generate ArUco:\n{e}")
            self.status_var.set(f"❌ Error: {e}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = QRGeneratorApp(root)
    root.mainloop()