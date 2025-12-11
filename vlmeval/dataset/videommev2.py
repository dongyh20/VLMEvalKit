from huggingface_hub import snapshot_download
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import ast

FAIL_MSG = 'Failed to obtain answer via API.'

def cal_relevance(scores):
    score_map_exponential = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    exp_score = score_map_exponential.get(correct_count, 0.0)
    linear_score = correct_count * 25.0
    return exp_score, linear_score

def cal_logic(scores, group_structure):
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break
    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"未知的group_structure_list: {group_structure_list}")
    logic_score = score_map.get(last_correct_idx + 1, 0.0)
    return logic_score


def get_final_rating(score_file):
    data = load(score_file)
    # final_rating = {}
    # final_rating["score"] = data['score'].mean()
    # return final_rating
    all_groups = [[] for _ in range((len(data) + 1) // 4)]
    final_rating = {
        "level_1": [],
        "level_2": [],
        "level_3": [],
        "relevance_score": [],
        "relevance_linear_score": [],
        "logic_score": [],
        "total": [],
    }
    second_head_rating = {}
    third_head_rating = {}
    for i in range(len(data)):
        level, group_type, group_structure, score, second_head, third_head = (
            data.loc[i, 'level'],
            data.loc[i, 'group_type'],
            data.loc[i, 'group_structure'],
            data.loc[i, 'score'],
            data.loc[i, 'second_head'],
            data.loc[i, 'third_head'],
        )
        all_groups[i // 4].append((level, group_type, group_structure, score, second_head, third_head))
    for group in all_groups:
        level, group_type, group_structure, second_head, third_head = int(group[-1][0]), group[-1][1], group[-1][2], group[-1][4], group[-1][5]
        scores = [item[3] for item in group]
        if group_type == '相关性':
            exp_score, linear_score = cal_relevance(scores)
            final_rating['relevance_score'].append(exp_score)
            final_rating['relevance_linear_score'].append(linear_score)
        elif group_type == '逻辑链':
            exp_score = cal_logic(scores, group_structure)
            final_rating['logic_score'].append(exp_score)
        else:
            raise ValueError(f'未知的group_type: {group_type}')
        final_rating[f'level_{level}'].append(exp_score)
        final_rating['total'].append(exp_score)
        if second_head not in second_head_rating:
            second_head_rating[second_head] = []
        second_head_rating[second_head].append(exp_score)
        if third_head not in third_head_rating:
            third_head_rating[third_head] = []
        third_head_rating[third_head].append(exp_score)
    for key in final_rating:
        final_rating[key] = sum(final_rating[key]) / len(final_rating[key]) if len(final_rating[key]) > 0 else 0.0
    for key in second_head_rating:
        second_head_rating[key] = sum(second_head_rating[key]) / len(second_head_rating[key]) if len(second_head_rating[key]) > 0 else 0.0
    for key in third_head_rating:
        third_head_rating[key] = sum(third_head_rating[key]) / len(third_head_rating[key]) if len(third_head_rating[key]) > 0 else 0.0
    return {'final_rating': final_rating, 'second_head_rating': second_head_rating, 'third_head_rating': third_head_rating}


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class VideoMMEv2(VideoBaseDataset):

    MD5 = '9dcf6f01c50b4e67addac2dcee855f1a'
    SYS = ''

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.
"""

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Video-MME-v2', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME-v2']

    def prepare_dataset(self, dataset_name='Video-MME-v2', repo_id=''):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, f"{video_pth:03d}" + '.mp4')):
                    return False
            return True

        # original usage
        # cache_path = get_cache_path(repo_id)
        cache_path = "/apdcephfs_jn/share_302244400/peterrao/data/videommev2"
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('video')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet("/mnt/castle/dyh/VLMEvalKit/test.parquet")
                data_file = data_file.assign(index=range(len(data_file)))

                data_file['video'] = data_file['video_id'].apply(lambda x: str(x))
                data_file = data_file[['index', 'video', 'url', 'group_type', 'group_structure',
                                       'question_id', 'question', 'options', 'answer', 'level', 'second_head', 'third_head', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            # original usage
            # if modelscope_flag_set():
            #     from modelscope import dataset_snapshot_download
            #     dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            # else:
            #     dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            # unzip_hf_zip(dataset_path)
            # generate_tsv(dataset_path)
            dataset_path = "/apdcephfs_jn/share_302244400/peterrao/data/videommev2"
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = osp.join(self.data_root, 'video', f"{video:03d}" + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(f"{video:03d}")
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(f"{video:03d}", len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', f"{line['video']:03d}" + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.FRAMES_TMPL_NOSUB
        message.append(dict(type='text', value=text_prompt))
        line['question'] += '\n' + line['options']
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import extract_characters_regex_v2, extract_option

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        # if not osp.exists(score_file):
        if True:
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex_v2(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'Video-MME'
                    )
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_characters_regex_v2(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)
        rating = get_final_rating(score_file)
        dump(rating, tgt_file)
        return rating
