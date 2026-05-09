# app.py

import streamlit as st
import tempfile
import os
from pose_logic import analyze_video

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="농구 AI 코치",
    page_icon="🏀",
    layout="wide"
)

# ── 헤더 ──────────────────────────────────────────────────────
st.title("🏀 농구 슈팅 AI 코칭 시스템")
st.caption("영상을 업로드하면 릴리즈 포인트의 팔꿈치·무릎 각도를 자동으로 분석합니다.")
st.divider()

# ── 사이드바 ───────────────────────────────────────────────────
with st.sidebar:
    st.header("분석 설정")

    release_threshold = st.slider(
        "릴리즈 감지 민감도 (픽셀)",
        min_value=40, max_value=120, value=80,
        help="손목-공 거리가 이 값 이상 벌어지면 릴리즈로 판정합니다."
    )

    shooting_hand = st.radio(
        "슈팅 손",
        ["오른손", "왼손"],
        index=0
    )

    st.divider()
    st.markdown("**이상적인 각도 기준**")
    st.markdown("- 팔꿈치: 85° ~ 105°")
    st.markdown("- 무릎: 150° 이상 (펴진 상태)")

# ── 메인 영역 ──────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("영상 업로드")
    uploaded_file = st.file_uploader(
        "슈팅 영상을 선택하세요",
        type=["mp4", "mov", "avi"],
        help="측면에서 촬영한 영상이 가장 정확합니다."
    )

    if uploaded_file:
        st.video(uploaded_file)  # 업로드한 영상 미리보기

with col_result:
    st.subheader("분석 결과")

    if not uploaded_file:
        st.info("왼쪽에서 영상을 업로드하면 여기에 결과가 표시됩니다.")

    else:
        # 분석 버튼
        if st.button("분석 시작", type="primary", use_container_width=True):

            # 임시 파일로 저장 (Streamlit은 파일 경로 필요)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("AI가 영상을 분석하는 중..."):
                try:
                    result = analyze_video(
                        video_path=tmp_path,
                        release_threshold=release_threshold,
                        use_left=(shooting_hand == "왼손")
                    )
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
                    result = None
                finally:
                    os.unlink(tmp_path)  # 임시 파일 삭제

            if result and result["detected"]:
                st.success("릴리즈 포인트 감지 완료!")

                # ── 릴리즈 순간 캡처 이미지 ────────────────
                if result.get("frame") is not None:
                    st.image(result["frame"], caption="릴리즈 포인트 순간", use_column_width=True)

                st.divider()

                # ── 팔꿈치 각도 카드 ───────────────────────
                c1, c2 = st.columns(2)

                elbow_angle = result["elbow_angle"]
                knee_angle  = result["knee_angle"]

                with c1:
                    color = "normal" if result["elbow_feedback"] == "Perfect!" else "inverse"
                    st.metric(
                        label="팔꿈치 각도",
                        value=f"{elbow_angle:.1f}°",
                        delta=result["elbow_feedback"]
                    )
                    if result["elbow_feedback"] == "Perfect!":
                        st.success("이상적인 릴리즈 각도입니다!")
                    else:
                        st.warning("팔꿈치를 더 굽혀 90도에 가깝게 유지하세요.")

                with c2:
                    st.metric(
                        label="무릎 각도",
                        value=f"{knee_angle:.1f}°",
                        delta=result["knee_feedback"]
                    )
                    if result["knee_feedback"] == "Good Leg!":
                        st.success("다리 신전이 잘 되었습니다!")
                    else:
                        st.warning("릴리즈 전 무릎을 더 활용해 점프력을 높이세요.")

                # ── 코칭 코멘트 ────────────────────────────
                st.divider()
                st.subheader("AI 코칭 코멘트")

                both_ok = (result["elbow_feedback"] == "Perfect!" and
                           result["knee_feedback"]  == "Good Leg!")

                if both_ok:
                    st.balloons()
                    st.success(
                        "훌륭한 슈팅 폼입니다! 팔꿈치와 무릎 각도 모두 이상적이에요. "
                        "이 폼을 꾸준히 유지하세요."
                    )
                elif result["elbow_feedback"] != "Perfect!":
                    st.warning(
                        f"팔꿈치 각도가 {elbow_angle:.0f}°로 측정됐습니다. "
                        "85~105° 범위를 목표로 연습하면 슛의 일관성이 높아집니다."
                    )
                else:
                    st.warning(
                        f"무릎 각도가 {knee_angle:.0f}°입니다. "
                        "릴리즈 순간 다리를 완전히 펴는 연습을 해보세요. "
                        "하체 힘이 슛 파워의 핵심입니다."
                    )

            elif result and not result["detected"]:
                st.error("릴리즈 포인트를 감지하지 못했습니다.")
                st.markdown("""
                **확인해보세요:**
                - 영상에서 농구공이 잘 보이는지
                - 슈터의 전신이 화면에 들어오는지
                - 사이드바의 민감도를 낮춰보세요 (40~60)
                """)