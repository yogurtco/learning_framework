from learning_framework.src.train.dataset.open_subtitles_data_set import OpenSubtitlesDataSet


def test_opensubtitles_dataset():
    dataset_path = "/data/opensubtitles/lines_en/test/"
    num_lines = 5
    dataset = OpenSubtitlesDataSet(
        data_path=dataset_path,
        num_lines=num_lines,
        preprocess=None,
    )
    for i in range(200000, 200013):
        item = dataset[i].text
        print(item)


if __name__ == '__main__':
    test_opensubtitles_dataset()
