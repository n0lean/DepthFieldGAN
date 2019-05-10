import glob
import os
import boto3
from skimage import transform, io
import tqdm
from boto.s3.connection import S3Connection

aws_s3_accessKey = "AKIA2IKMDTU4QDK7RR2B"
aws_s3_secretKey = "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"

save_path = '/Users/huxin/Desktop/adp/photo/focus/*.png'

s3 = boto3.client('s3', aws_access_key_id=aws_s3_accessKey,
                      aws_secret_access_key=aws_s3_secretKey)
bucket = 'photorefocus'

ls = [ ]

def loadImage():

	paths = glob.glob('/Users/huxin/Desktop/adp/photo/focus/*.png')
	for path in tqdm.tqdm(paths):
		img = io.imread(path)
		file_name = str(path.split('/')[-1])
		print(file_name)
		path = str(path)
		s3.upload_file(path, bucket, file_name)
		print("done")

def storeListObject():
	for key in s3.list_objects(Bucket=bucket)['Contents']:
		ls.append(str(key['Key']))
	ls.sort(key = lambda x: x.split(".")[0])
	print(ls[3])
	#lis = str(str(len(ls)) + " " + " ".join(map(str,ls)))
	#s3.put_object(Bucket=bucket,Key="list.txt",Body=lis)

if __name__ == '__main__':
	#loadImage()
	storeListObject()
