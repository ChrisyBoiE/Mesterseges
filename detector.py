from ultralytics import YOLO

class ObjektumDetektor:
    def __init__(self, modell_utvonal="yolov8s.pt", biztonsagi_kuszob=0.5, osztalyok=None, eszkoz="cpu"):
        self.modell = YOLO(modell_utvonal).to(eszkoz)
        self.biztonsagi_kuszob = biztonsagi_kuszob
        self.osztalyok = osztalyok  # Az osztályok listája, amelyeket detektálni szeretnénk

    def detektal(self, kep):
        eredmenyek = self.modell(kep, conf=self.biztonsagi_kuszob, classes=self.osztalyok)
        annotalt_kep = eredmenyek[0].plot()
        return annotalt_kep
