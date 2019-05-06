import appex
from PIL import Image

def main():
	if not appex.is_running_extension():
		print('Running in Pythonista app, using test image...')
		img = appex.get_image()
#		img = Image.open('test:Mandrill')
	else:
		img = appex.get_image()
	if img:
		# TODO: Your own logic here...
		print('Converting image to grayscale...')
		grayscale = img.convert('L')
		grayscale.show()
	else:
		print('No input image found')

if __name__ == '__main__':
	main()
