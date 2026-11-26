"""👨‍💻 About Developer & Contact — profile and contact form."""
import base64
import logging
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

import streamlit as st

from src import ui

st.set_page_config(page_title="About & Contact", page_icon="👨‍💻", layout="wide")
ui.inject()
st.title("👨‍💻 About & Contact")
st.caption("The developer behind the Micro-Credit Risk Hub and how to get in touch.")

NAME = "Ajeet Kumar"
EMAIL = "ajeetk095@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/ajeek095/"
GITHUB = "https://github.com/ajeetk095"
REPO = "https://github.com/ajeetk095/microcredit-risk-app"  # update if your repo name differs
PHOTO_PATH = Path(__file__).resolve().parent.parent / "assets" / "ajeet_img.png"  # developer photo


def avatar_html() -> str:
    """Developer photo from assets/ (embedded as base64), with an initials badge as fallback."""
    if PHOTO_PATH.exists():
        b64 = base64.b64encode(PHOTO_PATH.read_bytes()).decode()
        return (f'<img class="avatar" src="data:image/png;base64,{b64}" alt="{NAME}" '
                'style="width:150px;height:150px;min-width:150px;aspect-ratio:1/1;border-radius:50%;'
                'object-fit:cover;object-position:center;display:block;margin:0 auto .8rem auto;'
                'border:3px solid #1f6feb;box-shadow:0 2px 8px rgba(0,0,0,.25)">')
    initials = "".join(w[0] for w in NAME.split()[:2]).upper()
    return f'<div class="avatar-fallback">{initials}</div>'


profile_col, contact_col = st.columns(2, gap="large")

with profile_col:
    st.subheader("About the Developer")
    st.markdown(
        f"""<div class="info-card">{avatar_html()}<h3>{NAME}</h3>
        <div class="section-note">Quantitative Finance Professional<br>Bangalore, India</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Hi, I'm **{NAME}** — a quantitative finance professional based in Bangalore. I hold a "
        "**B.Tech in IT** (AKTU), an **M.Tech in CSE** (NSUT Delhi) and an **M.Sc. in Financial "
        "Engineering** from **WorldQuant University**.\n\n"
        "I apply quantitative methods and machine learning to real-world finance problems. The "
        "**Micro-Credit Risk Hub** is a worked example: it takes a telecom airtime-loan delinquency "
        "brief, builds a leakage-free model, converts probabilities into loan economics and "
        "approve / limit / decline decisions, and layers on a live market-stress view.\n\n"
        "I am seeking a full-time role in **quantitative research or financial engineering**."
    )
    b1, b2, b3 = st.columns(3)
    b1.link_button("📧 Email", f"mailto:{EMAIL}", width="stretch")
    b2.link_button("💼 LinkedIn", LINKEDIN, width="stretch")
    b3.link_button("🐙 GitHub", GITHUB, width="stretch")
    st.link_button("📦 View this project's source code", REPO, width="stretch")

with contact_col:
    st.subheader("Contact")
    st.caption("Send a message about opportunities, research collaborations, or feedback on this app.")

    with st.form("contact_form", clear_on_submit=False):
        name = st.text_input("Name")
        email = st.text_input("Email")
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=180)
        submitted = st.form_submit_button("Send Message", width="stretch")

    if submitted:
        errors = []
        if len(name.strip()) < 2:
            errors.append("Please enter your name (min. 2 characters).")
        if "\r" in name or "\n" in name:
            errors.append("Name cannot contain line breaks.")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
            errors.append("Please enter a valid email address.")
        if len(subject.strip()) < 3:
            errors.append("Subject must be at least 3 characters.")
        if "\r" in subject or "\n" in subject:
            errors.append("Subject cannot contain line breaks.")
        if len(message.strip()) < 10:
            errors.append("Message must be at least 10 characters.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                cfg = st.secrets.get("email", {})
                host = cfg.get("smtp_host", "smtp.gmail.com")
                port = int(cfg.get("smtp_port", 465))
                user, pwd = cfg.get("username", ""), cfg.get("password", "")
                to_addr = cfg.get("to_address", EMAIL)
                if not user or not pwd:
                    raise RuntimeError("Contact email credentials are not configured.")

                mail = EmailMessage()
                mail["From"] = user
                mail["To"] = to_addr
                mail["Reply-To"] = email.strip()
                mail["Subject"] = f"[Micro-Credit Risk Hub] {subject.strip()}"
                mail.set_content(
                    "New contact form message\n\n"
                    f"Name: {name.strip()}\nEmail: {email.strip()}\nSubject: {subject.strip()}\n\n"
                    f"Message:\n{message.strip()}"
                )
                with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                    smtp.login(user, pwd)
                    smtp.send_message(mail)
                st.success("✅ Your message was sent successfully. Thank you!")
            except Exception:
                logging.exception("Failed to send contact form email")
                st.error(f"We couldn't send your message right now. Please try again later or email {EMAIL} directly.")

    st.markdown(
        '<div class="section-note">Typical response time: within 24–48 hours. Please include the '
        "role or collaboration context in your message.</div>",
        unsafe_allow_html=True,
    )
