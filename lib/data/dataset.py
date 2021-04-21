import os
from torch.utils import data
import numpy as np



class HumansDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, mode, split=None, category='D-FAUST',
                 length_sequence=17, n_files_per_sequence=-1, offset_sequence=0,
                 ex_folder_name='pcl_seq', specific_model=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            category (str): category of data
            split (str): which split is used
        '''

        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.mode = mode
        self.length_sequence = length_sequence
        self.n_files_per_sequence = n_files_per_sequence
        self.offset_sequence = offset_sequence
        self.ex_folder_name = ex_folder_name

        if mode == 'train':
            with open(os.path.join(self.dataset_folder, 'train_human_ids.lst'), 'r') as f:
                self.hid = f.read().split('\n')

        if specific_model is not None:
            self.models = [{'category': category, 'model': specific_model['seq'],
                            'start_idx': specific_model['start_idx']}]
        else:
            # Get all models
            self.models = []
            subpath = os.path.join(self.dataset_folder, 'test', category)
            if split is not None and os.path.exists(
                    os.path.join(subpath, split + '.lst')):
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            else:
                models_c = [f for f in os.listdir(subpath) if
                            os.path.isdir(os.path.join(subpath, f))]
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            models_len = self.get_models_seq_len(subpath, models_c)
            models_c, start_idx = self.subdivide_into_sequences(
                models_c, models_len)
            self.models += [
                {'category': category, 'model': m, 'start_idx': start_idx[i]}
                for i, m in enumerate(models_c)
            ]


    def __len__(self):
        if self.mode == 'train':
            return 2048 * 16
        else:
            return len(self.models)


    def __getitem__(self, idx):
        '''identity enchange is conducted only in training mode
        '''
        if self.mode == 'train':
            id1, id2 = np.random.choice(self.hid, size=2, replace=False)
            m1, m2 = np.random.choice(self.models, size=2)
            data1 = self.get_data_dict(m1, idx, id1, id2)
            data2 = self.get_data_dict(m2, idx, id2, id1)
            return [data1, data2]
        else:
            m = self.models[idx]
            data = self.get_data_dict(m, idx)
            return data


    def get_data_dict(self, motion, idx=0, id1=None, id2=None):
        category = motion['category']
        model = motion['model']
        start_idx = motion['start_idx']

        data = {}

        for field_name, field in self.fields.items():
            if self.mode == 'train':
                if field_name in ['points_ex', 'points_t_ex', 'points_iou_ex']:
                    model_path = os.path.join(self.dataset_folder, 'train', category, id2, model)
                else:
                    model_path = os.path.join(self.dataset_folder, 'train', category, id1, model)
            else:
                model_path = os.path.join(self.dataset_folder, 'test', category, model)

            field_data = field.load(model_path, idx, start_idx=start_idx, dataset_folder=self.dataset_folder)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        return data


    def get_model_dict(self, idx):
        return self.models[idx]


    def get_models_seq_len(self, subpath, models):
        ''' Returns the sequence length of a specific model.

        This is a little "hacky" as we assume the existence of the folder
        self.ex_folder_name. However, in our case this is always given.

        Args:
            subpath (str): subpath of model category
            models (list): list of model names
        '''
        ex_folder_name = self.ex_folder_name
        models_seq_len = [
            len(os.listdir(os.path.join(subpath, m, ex_folder_name)))
            for m in models]

        return models_seq_len


    def subdivide_into_sequences(self, models, models_len):
        ''' Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        '''
        length_sequence = self.length_sequence
        n_files_per_sequence = self.n_files_per_sequence
        offset_sequence = self.offset_sequence

        # Remove files before offset
        models_len = [l - offset_sequence for l in models_len]

        # Reduce to maximum number of files that should be considered
        if n_files_per_sequence > 0:
            models_len = [min(n_files_per_sequence, l) for l in models_len]

        models_out = []
        start_idx = []
        for idx, model in enumerate(models):
            for n in range(0, models_len[idx] - length_sequence + 1):
                models_out.append(model)
                start_idx.append(n + offset_sequence)

        return models_out, start_idx
