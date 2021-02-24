from unittest import mock
from indicators.core.nlp_utils import (
    join_text,
    join_and_filter_sent,
    join_doc,
    vectorise_docs,
    parse_topic,
    parse_corex_topics,
)

PATH = "indicators.core.nlp_utils.{}"
TOPIC_1 = "topic1:this,is,a,topic,and,other,words"
TOPIC_2 = "topic2:this,is,another,topic,and,other,words"


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
    topic = parse_topic(TOPIC_1, n_most=5)
    assert topic == "this is a topic and"

    topic = parse_topic(TOPIC_1, n_most=4)
    assert topic == "this is a topic"


@mock.patch("builtins.open", mock.mock_open(read_data=f"{TOPIC_1}\n{TOPIC_2}"))
def test_parse_corex_topics():
    topics = parse_corex_topics("dummy", n_most=6)
    assert len(topics) == 2
    assert topics[0] == "this is a topic and other"
    assert topics[1] == "this is another topic and other"
