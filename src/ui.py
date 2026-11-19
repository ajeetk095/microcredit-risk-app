import streamlit as st

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1200px}
.rec{border:1px solid rgba(128,128,128,.25);border-left:6px solid var(--c);border-radius:10px;padding:14px 18px;margin-bottom:12px;background:rgba(128,128,128,.05)}
.rec h4{margin:0 0 6px 0;font-size:1.02rem}
.rec p{margin:0 0 6px 0;font-size:.93rem;line-height:1.45}
.rec small{opacity:.7}
.tag{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.04em;padding:2px 8px;border-radius:999px;color:#fff;background:var(--c);margin-right:8px;vertical-align:middle}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;color:#fff}
</style>"""
COLORS = {"High": "#c0392b", "Medium": "#d68910", "Info": "#2e86c1"}
REGIME = {"Calm": "#1e8449", "Elevated": "#d68910", "Stressed": "#c0392b"}


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def rec_card(r):
    st.markdown(
        f"<div class='rec' style='--c:{COLORS[r['priority']]}'><h4><span class='tag'>{r['priority'].upper()}</span>{r['title']}</h4>"
        f"<p>{r['body']}</p><small>{r['evidence']}</small></div>", unsafe_allow_html=True)


def pill(text, color):
    return f"<span class='pill' style='background:{color}'>{text}</span>"
