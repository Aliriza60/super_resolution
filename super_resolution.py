

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
	"--model_name",
	type=str,
	default="",
	help="Model Name",
)

parser.add_argument(
	"--image_path",
	type=str,
	default="",
	help="Model Name",
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
				--model_name: {ap.model_name} \
				--image_path: {ap.image_path} \
				--scale: {ap.scale}"

print(info_text)

image = Image.open(ap.image_path)

if ap.model_name == "edsr-base":

	model = EdsrModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "msrn-bam":

	model = MsrnModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "a2n":

	model = A2nModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "carn":

	model = CarnModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "drln-bam":

	model = DrlnModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "edsr":

	model = EdsrModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "mdsr":

	model = MdsrModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "msrn":

	model = MsrnModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "pan-bam":

	model = PanModel.from_pretrained(ap.model_path, scale=ap.scale)

elif ap.model_name == "pan":

	model = PanModel.from_pretrained(ap.model_path, scale=ap.scale)


inputs = ImageLoader.load_image(image)
preds = model(inputs)
os.makedirs(ap.output, exist_ok=True)

ImageLoader.save_image(preds, f""+ap.output+"/"+"scaled"+".png") 

