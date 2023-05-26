import os
import requests
import shutil
from tqdm.auto import tqdm
from zipfile import ZipFile


class NLVR2_DATA(object):
     def download(self):
        img_zip = os.path.join(self.img_temp_abs_path, "test1_img.zip")
        if(not os.path.exists(self.img_temp_abs_path)):
            with requests.get(self.img_zip_url, stream=True) as r:
                
                # check header to get content length, in bytes
                total_length = int(r.headers.get("Content-Length"))
                
                # implement progress bar via tqdm
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:
                
                    # save the output to a file
                    with open(img_zip, 'wb')as output:
                        shutil.copyfileobj(raw, output)
        else:
            print("img zip ready:", img_zip)

        # extract file into self.img_root
        
        if(not os.path.exists(os.path.join(self.img_root, 'test1'))):
            print("extacting into " + self.img_root)
            with ZipFile(img_zip, 'r') as zObject:
                zObject.extractall(path=str(self.img_root))
        else:
            print("images ready: ", os.path.join(self.img_root, 'test1'))
        
     def __init__(self):
        self.img_root_abs = ""
        file_abs = os.path.abspath(__file__)
        self.repo_abs_path = os.path.sep.join(file_abs.split(os.path.sep)[:-3])

        self.submodule_path = "nlvr_data_submodule"
        self.test_json_path = "nlvr_test.json"
        self.submodule_abs_path = os.path.join(self.repo_abs_path, self.submodule_path)
        self.test_json_abs_path = os.path.join(self.repo_abs_path, self.test_json_path)
        self.img_root = os.path.join(self.repo_abs_path, "nlvr2_images_4_test2json")
        self.img_root_abs = self.img_root
        self.img_zip_url = "https://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip"
        self.img_temp_abs_path = os.path.join(self.repo_abs_path, "temp_nlvr2_images_4_test2json")
        if(not os.path.exists(self.img_temp_abs_path)):
            os.mkdir(self.img_temp_abs_path)

d = NLVR2_DATA()
d.download()
print(d.__dict__)
       