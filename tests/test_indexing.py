import pytest

from pneumonia.indexing import (
    DOWNLOAD_CMD,
    DatasetNotFoundError,
    build_index,
    count_table,
    find_dataset_root,
    parse_filename,
    parse_patient_id,
)


@pytest.mark.parametrize(
    "name, pid, subtype",
    [
        ("person8_bacteria_37.jpeg", "person8", "bacteria"),
        ("person8_bacteria_38.jpeg", "person8", "bacteria"),
        ("person1_virus_6.jpeg", "person1", "virus"),
        ("person1000_virus_1681.jpeg", "person1000", "virus"),
        ("IM-0115-0001.jpeg", "IM-0115", "normal"),
        ("NORMAL2-IM-0222-0001.jpeg", "NORMAL2-IM-0222", "normal"),
        ("NORMAL2-IM-0222-0002.jpeg", "NORMAL2-IM-0222", "normal"),
    ],
)
def test_parse_real_looking_names(name, pid, subtype):
    assert parse_filename(name) == (pid, subtype)


def test_same_child_groups_together():
    assert parse_patient_id("person8_bacteria_37.jpeg") == parse_patient_id(
        "person8_bacteria_38.jpeg"
    )
    assert parse_patient_id("person8_bacteria_37.jpeg") != parse_patient_id(
        "person80_bacteria_1.jpeg"
    )
    assert parse_patient_id("IM-0222-0001.jpeg") != parse_patient_id("NORMAL2-IM-0222-0001.jpeg")


def test_unparseable_name_gets_unique_id_and_warns():
    with pytest.warns(UserWarning):
        pid = parse_patient_id("weird file.jpeg")
    assert pid.startswith("UNPARSED-")


def test_find_root_resolves_nested_layout(synth_tree):
    root = find_dataset_root(synth_tree)
    assert root.name == "chest_xray" and root.parent.name == "chest_xray"
    assert (root / "train" / "NORMAL").is_dir()


def test_missing_dataset_error_names_path_and_command(tmp_path):
    with pytest.raises(DatasetNotFoundError) as e:
        find_dataset_root(tmp_path / "nope")
    assert DOWNLOAD_CMD in str(e.value)
    assert "nope" in str(e.value)


def test_index_and_counts(synth_tree):
    df = build_index(find_dataset_root(synth_tree))
    assert set(df.columns) >= {"path", "class_name", "label", "official_split", "patient_id"}
    assert len(df) == 20
    t = count_table(df)
    assert t.loc["train", "images_NORMAL"] == 6
    assert t.loc["train", "patients_PNEUMONIA"] == 3  # 2 images per synthetic patient
