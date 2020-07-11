from torchvision.transforms import Compose

from learning_framework.src.train.dataset.open_subtitles_data_set import OpenSubtitlesDataSet
from learning_framework.src.train.preprocess.character_embedding_preprocess import CharacterEmbeddingPreprocess


def test_opensubtitles_dataset():
    dataset_path = "/data/opensubtitles/lines_en/test/"
    num_lines = 5
    data_transform = Compose([CharacterEmbeddingPreprocess()])

    dataset = OpenSubtitlesDataSet(
        data_path=dataset_path,
        num_lines=num_lines,
        preprocess=data_transform,
    )

    # try getting the last item
    assert dataset[len(dataset)-1]


if __name__ == '__main__':
    test_opensubtitles_dataset()
