

# python super_resolution.py --model_path mdsr --output output --scale 4

import argparse
import os
from super_image import EdsrModel, ImageLoader
from super_image import MsrnModel
from super_image import A2nModel
from super_image import CarnModel
from super_image import DrlnModel
from super_image import EdsrModel
from super_image import MdsrModel
from super_image import MsrnModel
from super_image import PanModel
from PIL import Image

def list_of_strings(arg):
	 return arg.split(',')

parser = argparse.ArgumentParser()

parser.add_argument(
	"--workerTaskId",
	type=int,
	default=0,
	help="Worker Task Id"
)

parser.add_argument(
	"--model_path",
	type=str,
	default="",
	help="Model Path",
)

parser.add_argument(
	"--output",
	type=str,
	default="",
	help="Output Folder",
)

parser.add_argument(
	"--scale",
	type=int,
	default=0,
	help="Scale"
)

ap = parser.parse_args()

info_text = f"Script Version : 1.1 \
				Arguments : \
				--workerTaskId: {ap.workerTaskId} \
				--output: {ap.output} \
				--model_path: {ap.model_path} \
				--scale: {ap.scale}"

print(info_text)

image = Image.open("example_1.jpg")

if ap.model_path == "edsr-base":

	model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=ap.scale)

elif ap.model_path == "msrn-bam":

	model = MsrnModel.from_pretrained("eugenesiow/msrn-bam", scale=ap.scale)

elif ap.model_path == "a2n":

	model = A2nModel.from_pretrained('eugenesiow/a2n', scale=ap.scale)

elif ap.model_path == "carn":

	model = CarnModel.from_pretrained('eugenesiow/carn', scale=ap.scale)

elif ap.model_path == "drln-bam":

	model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=ap.scale)

elif ap.model_path == "edsr":

	model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=ap.scale)

elif ap.model_path == "mdsr":

	model = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=ap.scale)

elif ap.model_path == "msrn":

	model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=ap.scale)

elif ap.model_path == "pan-bam":

	model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=ap.scale)

elif ap.model_path == "pan":

	model = PanModel.from_pretrained('eugenesiow/pan', scale=ap.scale)


inputs = ImageLoader.load_image(image)
preds = model(inputs)
os.makedirs(ap.output, exist_ok=True)

ImageLoader.save_image(preds, f""+ap.output+"/"+"scaled"+".png") 

