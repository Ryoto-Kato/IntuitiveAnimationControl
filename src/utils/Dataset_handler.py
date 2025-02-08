"""
Generic dataset handler
"""

import os
import sys

class Filehandler:
    def __init__(self, path_to_dataset, ext = '.obj', prefix = 'E0'):
        self.path_to_dataset = path_to_dataset
        self.list_dirs = []
        self.list_expNames = []
        self.list_expPathFiles = []
        self.dict_objs = {}
        self.num_exps = 0
        self.ext = ext
        self.prefix = prefix
    
    def get_path_to_dataset(self):
        return self.path_to_dataset

    def iter_dir(self):
        self.num_exps = 0
        for i, name in enumerate(os.listdir(self.path_to_dataset)):
            f = os.path.join(self.path_to_dataset, name)
            if os.path.isdir(f) and name.startswith(self.prefix):
                # print(f'{i}, {name}')
                self.list_expNames.append(name)
                self.list_dirs.append(f)
                self.list_expPathFiles.append(f)
                self.num_exps = self.num_exps + 1
                
        # print("Directory path")
        self.list_expNames.sort()
        self.list_dirs.sort()
        self.list_expPathFiles.sort()
        # print(self.list_dirs)
        
            
        for i, dir in enumerate(self.list_dirs):
            # print(dir)
            list_objs = []
            for j, file_name in enumerate(os.listdir(os.path.join(dir))):
                if file_name.endswith(self.ext):
                    # print(file_name)
                    list_objs.append(file_name)                    
                else:
                    continue
            list_objs.sort()
            self.dict_objs.update({i: list_objs})

    @staticmethod
    def dirwalker_InFolder(path_to_folder, prefix):
        """
        Arguments
            path_to_folder: str
            prefix: the beginning of target folder name

        Returns
            list_dirNames, list_dirPaths: list of directory name, list of directory path
        """
        list_dirNames = []
        list_dirPaths = []
        for i, name in enumerate(os.listdir(path_to_folder)):
            f = os.path.join(path_to_folder, name)
            if os.path.isdir(f) and prefix!= None and name.startswith(prefix):
                list_dirNames.append(name)
                list_dirPaths.append(f)
                # print(f'{i}, {name}')
        list_dirNames.sort()
        list_dirPaths.sort()
        return list_dirNames, list_dirPaths
    
    @staticmethod
    def fileWalker_InDirectory(path_to_directory, ext):
        """
        Arguments
            path_to_directory: str,
            ext: the name of target extension

        Returns
            list_fileNames, list_filePaths: list of file name, list of file path
        """

        list_filePaths = []
        list_fileNames = []
        for i, f_name in enumerate(os.listdir(path_to_directory)):
            if f_name.endswith(ext):
                list_fileNames.append(f_name)
                list_filePaths.append(os.path.join(path_to_directory, f_name))
            else:
                continue
        list_fileNames.sort()
        list_filePaths.sort()


        return list_fileNames, list_filePaths
            