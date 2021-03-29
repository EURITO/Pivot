from unittest import mock
from indicators.core.nlp_utils import (
    join_text,
    join_and_filter_sent,
    join_doc,
    vectorise_docs,
    parse_topic,
    parse_corex_topics,
    parse_corex_paths,
    get_corex_labels,
    get_non_stop_topics,
    get_antitopics,
    get_fluffy_topics,
    parse_clean_topics,
)

PATH = "indicators.core.nlp_utils.{}"
TOPIC_1 = "topic1:this,is,a,topic,and,other,words"
TOPIC_2 = "topic2:this,is,another,topic,and,other,words"
COVID_TOPIC = "covid_19 sars_cov_2 coronavirus disease patients"


def test_join_text():
    joined = join_text("this", "is", None, "some", None, "text")
    assert joined == "this is some text"


def test_join_doc_no_stops():
    doc = [
        ("this", "is", "some", None, "cool", "text"),
        ("this", "is", None, "more", "cool", None, "text", "again"),
    ]
    processed = join_doc(doc, extra_stops=[])
    assert processed == "this is some cool text\nthis is more cool text again"


def test_join_doc_stops():
    doc = [
        ("this", "is", "some", None, "cool", "text"),
        ("this", "is", None, "more", "cool", None, "text", "again"),
    ]
    processed = join_doc(doc, extra_stops=["more", "cool", "text"])
    assert processed == "this is some\nthis is again"


def test_join_and_filter_sent():
    sent = ["this", "sent", "is", None, "a", "good", "sent"]
    assert join_and_filter_sent(sent, []) == "this sent is a good sent"
    assert join_and_filter_sent(sent, ["good", "sent"]) == "this is a"


@mock.patch(PATH.format("Ngrammer"))
def test_vectorise_docs(mocked_Ngrammer):

    # Inputs and outputs
    ngrammed_doc = [["this", "is", "a", "sent"], ["this", "is", "another", "sent"]]
    expected_features = ["another", "is", "sent", "this"]
    expected_count_vec = [[1, 2, 2, 2]]

    # Mock up the ngrammer
    mocked_ngrammer = mock.Mock()
    mocked_ngrammer.process_document.return_value = ngrammed_doc
    mocked_Ngrammer.return_value = mocked_ngrammer

    # Run the test
    vectors, features = vectorise_docs(["dummy"], min_df=1, max_df=1.0)
    assert (vectors.todense() == expected_count_vec).all()
    assert features == expected_features


def test_parse_topic():
    topic = parse_topic(TOPIC_1)
    assert topic == "this is a topic and"


@mock.patch("builtins.open", mock.mock_open(read_data=f"{TOPIC_1}\n{TOPIC_2}"))
@mock.patch(PATH.format("parse_corex_paths"))
def test_parse_corex_topics(mocked_parser):
    topics = parse_corex_topics("dummy")
    assert len(topics) == 2
    assert topics[0] == "this is a topic and"
    assert topics[1] == "this is another topic and"


def test_parse_corex_paths():
    from indicators.core.tests import dummy_topic_module

    paths = parse_corex_paths(dummy_topic_module)
    assert set(paths.keys()) == {"labels", "topics", "most_deterministic_groups"}


def test_get_corex_labels():
    """Check that can read the topics data and that the shape is as expected"""
    from indicators.core.tests import dummy_topic_module as mod

    # i.e. check cache is working
    assert get_corex_labels(mod) is get_corex_labels(mod)
    assert get_corex_labels(mod).shape == (10, 150)
    assert get_corex_labels(mod).columns[0] == COVID_TOPIC
    assert get_corex_labels(mod).sum(axis=1).sum() == 296


def test_get_non_stop_topics():
    from indicators.core.tests import dummy_topic_module

    non_stops = get_non_stop_topics(dummy_topic_module)
    # allow the config to change, but it shouldn't change the results too much
    assert len(non_stops) > 50 and len(non_stops) < 296
    assert COVID_TOPIC in non_stops


def test_get_antitopics():
    from indicators.core.tests import dummy_topic_module

    antitopics = get_antitopics(dummy_topic_module)
    assert len(antitopics) > 1 and len(antitopics) < 15
    assert COVID_TOPIC not in antitopics


def test_get_fluffy_topics():
    from indicators.core.tests import dummy_topic_module

    fluffy_topics = get_fluffy_topics(dummy_topic_module)
    assert len(fluffy_topics) > 30 and len(fluffy_topics) < 80
    assert COVID_TOPIC not in fluffy_topics


def test_parse_clean_topics():
    from indicators.core.tests import dummy_topic_module

    clean_topics = parse_clean_topics(dummy_topic_module)
    rows, cols = clean_topics.shape
    assert rows == 10
    assert cols > 20 and cols < 100
    assert COVID_TOPIC in clean_topics.columns
