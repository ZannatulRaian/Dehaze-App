"""Download real, clear city-street photos for the urban evaluation (they are NOT bundled with the project).

    python get_street_photos.py

Fetches 8 street photos with vehicles from the image folder of github.com/matterport/Mask_RCNN (the repository is
MIT-licensed; the photos come from Flickr and keep their own licences, so they are only downloaded for local testing,
not redistributed here) and copies the bundled `assets/test images/2467248.jpg`. Everything goes to
`assets/test images/urban_real_clear/`. `evaluate.py` fogs these clear photos (rough depth guess) and measures how well
the pipeline restores them, including a vehicle and pedestrian detection test (see detect.py).
"""
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "test images" / "urban_real_clear"
BASE = "https://raw.githubusercontent.com/matterport/Mask_RCNN/master/images/"
PHOTOS = ["12283150_12d37e6389_z.jpg", "25691390_f9944f61b5_z.jpg", "7933423348_c30bd9bd4e_z.jpg",
          "8053677163_d4c8f416be_z.jpg", "8239308689_efa6c11b08_z.jpg", "8512296263_5fc5458e20_z.jpg",
          "8699757338_c3941051b6_z.jpg", "9118579087_f9ffa19e63_z.jpg"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = ROOT / "assets" / "test images" / "2467248.jpg"
    if local.exists():
        shutil.copy(local, OUT / "street00_2467248.jpg")
    for i, name in enumerate(PHOTOS, 1):
        dest = OUT / f"street{i:02d}_{name}"
        if not dest.exists():
            urllib.request.urlretrieve(BASE + name, dest)
        print("ok", dest.name)
    print(f"{len(list(OUT.glob('*.jpg')))} photos in {OUT}")


if __name__ == "__main__":
    main()
