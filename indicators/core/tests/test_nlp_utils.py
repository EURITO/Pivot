from unittest import mock
from indicators.core.nlp_utils import join_text, join_doc, vectorise_docs

PATH = "indicators.core.nlp_utils.{}"


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


@mock.patch(PATH.format("Ngrammer"))
def test_vectorise_docs(mocked_Ngrammer):
    ngrammed_doc = [["this", "is", "a", "sent"], ["this", "is", "another", "sent"]]
    mocked_ngrammer = mock.Mock()
    mocked_ngrammer.process_document.return_value = ngrammed_doc
    mocked_Ngrammer.return_value = mocked_ngrammer
    processed = vectorise_docs(["dummy"], 0, 1)
    print(processed)
