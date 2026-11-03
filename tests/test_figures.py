import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image, UnidentifiedImageError

from paper_radar import app, figures
from paper_radar.feishu import FeishuImageUploader, build_digest_card
from paper_radar.models import DeepRead, MatchResult, Paper, PaperFigure, Recommendation


def paper(paper_id="2611.00928"):
    now = datetime.now(UTC)
    return Paper(paper_id, "Excitons", (), "Exciton dispersion in hBN", now, now, (),
                 f"https://arxiv.org/abs/{paper_id}", f"https://arxiv.org/pdf/{paper_id}")


def png():
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, format="PNG")
    return buf.getvalue()


def source():
    return figures.SourceFigure("S1.F2", "Figure 2", "Figure 2: measured spectrum",
                                "https://arxiv.org/html/2611.00928#S1.F2",
                                ("https://arxiv.org/html/2611.00928/fig2.png",))


def prepared():
    return figures.PreparedFigure(PaperFigure(
        "Figure 2", "对比两条曲线，查看激子峰随栅压的移动。", source().source_url,
        source().image_urls,
    ), (png(),))


def recommendation(p=None):
    return Recommendation(p or paper(), 3, "检查本组的峰位分析。", ("激子",),
                          "激子峰位", "比较不同栅压的光谱。", True)


def test_html_keeps_all_panels_and_ignores_nested_figures_and_equation_images():
    html = '''<figure id="S1.F2" class="ltx_figure">
      <figure id="panel-a" class="ltx_figure"><img class="ltx_graphics" src="id/a.png">
      <figcaption>(a)</figcaption></figure>
      <figure id="panel-b" class="ltx_figure"><img class="ltx_graphics" src="id/b.png">
      <figcaption>(b)</figcaption></figure>
      <figcaption><span class="ltx_tag_figure">Figure 2:</span>
      A comparison <img class="ltx_graphics" src="equation.png">
      <math>x<annotation>xduplicate</annotation></math></figcaption>
      </figure><figure class="ltx_table"><figcaption>Table 1</figcaption></figure>'''
    result = figures.parse_figures(html, "https://arxiv.org/html/id")
    assert len(result) == 1
    assert result[0].figure_id == "S1.F2"
    assert result[0].image_urls == (
        "https://arxiv.org/html/id/a.png", "https://arxiv.org/html/id/b.png",
    )
    assert "duplicate" not in result[0].caption


@pytest.mark.parametrize("replacement", [
    "http://127.0.0.1/a.png", "https://other.test/a.png", "data:image/png;base64,aaa",
])
def test_html_rejects_external_images(replacement):
    html = f'''<figure id="f" class="ltx_figure"><img class="ltx_graphics"
        src="{replacement}"><figcaption><span class="ltx_tag_figure">Figure 1</span>
        Result</figcaption></figure>'''
    assert figures.parse_figures(html, "https://arxiv.org/html/2611.00928") == []


def test_arxiv_identity_supports_enriched_sources_and_exact_version():
    assert figures.arxiv_html_url(replace(paper("2611.00928v2"), source="arXiv + OpenAlex")) == (
        "https://arxiv.org/html/2611.00928v2"
    )
    assert figures.arxiv_html_url(replace(paper(), abstract_url="https://example.test/paper",
                                         pdf_url="https://example.test/paper.pdf")) is None


def test_source_link_is_pinned_to_the_image_version():
    html = '''<figure id="f" class="ltx_figure"><img class="ltx_graphics"
      src="2611.00928v2/fig.png"><figcaption><span class="ltx_tag_figure">Figure 1</span>
      Result</figcaption></figure>'''
    result = figures.parse_figures(html, "https://arxiv.org/html/2611.00928")
    assert result[0].source_url == "https://arxiv.org/html/2611.00928v2#f"
    assert figures.parse_figures(html, "https://arxiv.org/html/2611.00928v1") == []


def test_download_refuses_redirect_before_contacting_external_host(monkeypatch):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/secret"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(figures.httpx, "Client", lambda **kw: client)
    with pytest.raises(ValueError, match="supported arXiv"):
        figures._fetch("https://arxiv.org/html/2611.00928", limit=100)
    assert len(calls) == 1


def test_download_enforces_byte_limit(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x" * 101),
    ))
    monkeypatch.setattr(figures.httpx, "Client", lambda **kw: client)
    with pytest.raises(ValueError, match="budget"):
        figures._fetch("https://arxiv.org/html/2611.00928", limit=100)


def test_no_useful_figure_does_not_download_images(monkeypatch):
    monkeypatch.setattr(figures, "fetch_figures", lambda p: [source()])
    model = SimpleNamespace(_request_content_once=lambda *a, **kw: '{"figure_id":null}')
    monkeypatch.setattr(figures, "_fetch", lambda *a, **kw: pytest.fail("Unexpected image GET"))
    assert figures.prepare_key_figure(paper(), model) is None


def test_model_cannot_invent_a_figure(monkeypatch):
    monkeypatch.setattr(figures, "fetch_figures", lambda p: [source()])
    model = SimpleNamespace(_request_content_once=lambda *a, **kw: '{"figure_id":"invented"}')
    with pytest.raises(ValueError, match="absent"):
        figures.prepare_key_figure(paper(), model)


@pytest.mark.parametrize("useful", [True, False, "true"])
def test_image_is_visually_checked_before_it_can_be_attached(monkeypatch, useful):
    monkeypatch.setattr(figures, "fetch_figures", lambda p: [source()])
    monkeypatch.setattr(figures, "_fetch", lambda *a, **kw: (png(), source().image_urls[0]))
    requests = []

    def request(content, **kwargs):
        requests.append(content)
        if len(requests) == 1:
            return '{"figure_id":"S1.F2"}'
        return json.dumps({"useful": useful, "explanation_zh": "对比曲线，看激子峰的移动。"})

    result = figures.prepare_key_figure(paper(), SimpleNamespace(_request_content_once=request))
    assert requests[1][1]["type"] == "input_image"
    assert requests[1][1]["image_url"].startswith("data:image/png;base64,")
    assert (result is not None) is (useful is True)
    if result:
        with Image.open(io.BytesIO(result.images[0])) as image:
            assert image.size == (400, 300)


def test_invalid_image_is_rejected_before_vision():
    with pytest.raises(UnidentifiedImageError):
        figures._png(b"<html>not an image</html>")


@pytest.mark.parametrize("dry_run", [True, False])
def test_attach_limits_digest_to_one_figure_and_preserves_deep_read(monkeypatch, dry_run):
    monkeypatch.setenv("PAPER_FIGURES_ENABLED", "true")
    monkeypatch.setenv("FEISHU_IMAGE_APP_ID", "test")
    monkeypatch.setenv("FEISHU_IMAGE_APP_SECRET", "test")
    seen = []
    uploads = []
    recs = [recommendation(paper("2611.00001")), recommendation()]
    deep = DeepRead(paper(), "标题", "理由", "概括", ("方法",), ("结果",),
                    ("创新",), ("局限",), ("启发",))

    def prepare(p, model):
        seen.append(p.paper_id)
        return prepared()

    def upload(self, images):
        uploads.append(images)
        return ("img_test",)

    monkeypatch.setattr(app, "prepare_key_figure", prepare)
    monkeypatch.setattr(FeishuImageUploader, "upload", upload)
    result, updated_deep = app._attach_key_figure(recs, deep, object(), dry_run=dry_run)
    assert seen == [paper().paper_id]
    assert result[0].figure is None
    assert result[1].figure == updated_deep.figure
    assert len(uploads) == (0 if dry_run else 1)
    assert bool(updated_deep.figure.image_keys) is (not dry_run)
    assert deep.figure is None  # Do not mutate cached evaluations.


def test_upload_failure_keeps_text_and_original_figure_link(monkeypatch):
    monkeypatch.setenv("FEISHU_IMAGE_APP_ID", "test")
    monkeypatch.setenv("FEISHU_IMAGE_APP_SECRET", "test")
    monkeypatch.setattr(app, "prepare_key_figure", lambda *a: prepared())

    def fail(*a):
        raise RuntimeError("upload unavailable")

    monkeypatch.setattr(FeishuImageUploader, "upload", fail)
    rec = recommendation()
    result, _ = app._attach_key_figure([rec], None, object(), dry_run=False)
    card = build_digest_card(result, {rec.paper.paper_id: MatchResult(3, ("hBN",), ())})
    assert rec.summary_zh in str(card)
    assert source().source_url in str(card)
    assert "img_key" not in str(card)


def test_figure_failure_never_blocks_text(monkeypatch):
    def fail(*a):
        raise ValueError("unsupported source")

    monkeypatch.setattr(app, "prepare_key_figure", fail)
    recs = [recommendation()]
    assert app._attach_key_figure(recs, None, object(), dry_run=False) == (recs, None)


def test_card_places_figure_under_the_matching_paper():
    rec = replace(recommendation(), figure=replace(prepared().figure, image_keys=("img_test",)))
    card = build_digest_card([rec], {rec.paper.paper_id: MatchResult(3, ("hBN",), ())})
    elements = card["body"]["elements"][0]["elements"]
    image_index = next(i for i, e in enumerate(elements) if e["tag"] == "img")
    assert rec.title_zh in str(elements[image_index - 1])
    assert source().source_url in str(elements[image_index + 1])
    assert rec.figure.explanation_zh in str(elements[image_index + 1])


@pytest.mark.parametrize("body", [{"code": 0, "data": {}}, {"data": {"image_key": "bad"}}])
def test_upload_requires_success_code_and_image_key(monkeypatch, body):
    def handle(request):
        if "tenant_access_token" in request.url.path:
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "test-token"})
        assert b'image_type' in request.content
        assert b'image/png' in request.content
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "Client", lambda **kw: client)
    with pytest.raises(RuntimeError, match="acknowledge"):
        FeishuImageUploader("test", "test").upload((png(),))
