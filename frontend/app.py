
import hashlib

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("📚 ChatDoc AI")

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

if "last_upload_attempt" not in st.session_state:
    st.session_state.last_upload_attempt = None


# ---------------- DOCUMENT UPLOAD ----------------

st.write("Ask questions about your uploaded documents.")

uploaded_files = st.file_uploader(
    "📄 Choose PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    key="document_uploader",
    accept_multiple_files=True,
)

# Upload only when the button is clicked
if uploaded_files and st.button("Start Uploading", type="primary"):

    # Create a fingerprint for the selected batch
    batch_hash = hashlib.sha256()

    for uploaded_file in uploaded_files:
        batch_hash.update(uploaded_file.name.encode("utf-8"))
        batch_hash.update(uploaded_file.getvalue())

    upload_id = batch_hash.hexdigest()

    if upload_id != st.session_state.last_upload_attempt:

        # Mark the attempt before sending the request
        st.session_state.last_upload_attempt = upload_id

        files_to_upload = []

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.lower()

            if filename.endswith(".pdf"):
                file_type = "application/pdf"

            elif filename.endswith(".docx"):
                file_type = (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )

            else:
                file_type = "text/plain"

            files_to_upload.append(
                (
                    "files",
                    (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        file_type,
                    ),
                )
            )

        try:
            with st.spinner(
                f"Uploading and indexing {len(uploaded_files)} file(s)..."
            ):
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_to_upload,
                    timeout=300,
                )

                response.raise_for_status()
                result = response.json()

            st.session_state.document_uploaded = True

            st.session_state.uploaded_filenames = result.get(
                "filenames",
                [file.name for file in uploaded_files],
            )

            st.success(
                result.get(
                    "message",
                    "Documents uploaded and indexed successfully!",
                )
            )

            st.rerun()

        except requests.RequestException as e:
            # Allow retrying after a failed request
            st.session_state.last_upload_attempt = None
            st.error(f"Upload failed: {e}")

    else:
        st.info("These files have already been uploaded in this session.")


# ---------------- CHAT INTERFACE ----------------

if st.session_state.document_uploaded:

    st.divider()
    st.header("💬 Chat with your documents")

    st.caption(
        "Indexed documents: "
        + ", ".join(st.session_state.uploaded_filenames)
    )

    for chat in st.session_state.messages:

        role = (
            "user" if chat["role"] == "human"
            else "assistant" if chat["role"] == "ai"
            else chat["role"]
        )

        with st.chat_message(role):
            st.markdown(chat["content"])

    if prompt := st.chat_input(
        "Ask a question about your documents..."
    ):

        st.session_state.messages.append(
            {"role": "human", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):

                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"text": prompt},
                        timeout=180,
                    )

                    response.raise_for_status()
                    answer = response.json()["answer"]

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "ai", "content": answer}
                    )

                except requests.RequestException as e:
                    st.error(f"Backend error: {e}")

                except (KeyError, ValueError):
                    st.error("Unexpected response from the backend.")

else:
    st.info("📌Upload your documents first to start chatting.")
