"""
Unit tests for earnings_backtest.py
"""
import pytest

from earnings_backtest import (
    EarningsBacktest,
    infer_earnings_session_from_transcript,
)


class TestInferEarningsSessionFromTranscript:
    """Tests for infer_earnings_session_from_transcript function."""

    def test_only_good_morning_returns_bmo(self):
        """只有 good morning 應該回傳 BMO"""
        transcript = "Good morning, everyone. Welcome to our Q4 earnings call."
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

    def test_only_good_afternoon_returns_amc(self):
        """只有 good afternoon 應該回傳 AMC"""
        transcript = "Good afternoon, everyone. Welcome to our Q4 earnings call."
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_only_good_evening_returns_amc(self):
        """只有 good evening 應該回傳 AMC"""
        transcript = "Good evening, everyone. Welcome to our Q4 earnings call."
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_morning_before_afternoon_returns_bmo(self):
        """good morning 在 good afternoon 之前，應該回傳 BMO"""
        transcript = (
            "Good morning, everyone. Welcome to our call. "
            "As John said this good afternoon, we had a great quarter."
        )
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

    def test_afternoon_before_morning_returns_amc(self):
        """good afternoon 在 good morning 之前，應該回傳 AMC"""
        transcript = (
            "Good afternoon, everyone. Welcome to our call. "
            "Our CEO said good morning to the team earlier today."
        )
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_evening_before_morning_returns_amc(self):
        """good evening 在 good morning 之前，應該回傳 AMC"""
        transcript = (
            "Good evening, everyone. Welcome to our call. "
            "We started good morning procedures at 6 AM."
        )
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_no_greeting_returns_unknown(self):
        """沒有任何問候語應該回傳 UNKNOWN"""
        transcript = "Welcome to our Q4 earnings call. Let's begin with the results."
        assert infer_earnings_session_from_transcript(transcript) == "UNKNOWN"

    def test_empty_transcript_returns_unknown(self):
        """空字串應該回傳 UNKNOWN"""
        assert infer_earnings_session_from_transcript("") == "UNKNOWN"

    def test_none_transcript_returns_unknown(self):
        """None 應該回傳 UNKNOWN"""
        assert infer_earnings_session_from_transcript(None) == "UNKNOWN"  # type: ignore

    def test_case_insensitive(self):
        """應該不區分大小寫"""
        transcript = "GOOD MORNING everyone. Welcome to the call."
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

        transcript = "gOoD aFtErNoOn everyone."
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_all_three_greetings_earliest_wins(self):
        """三個問候語都有時，最早的那個贏"""
        # Morning first
        transcript = "Good morning! Good afternoon! Good evening!"
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

        # Afternoon first
        transcript = "Good afternoon! Good morning! Good evening!"
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

        # Evening first
        transcript = "Good evening! Good morning! Good afternoon!"
        assert infer_earnings_session_from_transcript(transcript) == "AMC"

    def test_greeting_in_middle_of_text(self):
        """問候語在文字中間也要能偵測到"""
        transcript = (
            "Operator: I would now like to turn the call over to the CEO. "
            "CEO: Good morning, everyone, and welcome to our call."
        )
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

    def test_real_world_transcript_snippet_bmo(self):
        """真實 transcript 片段測試 - 盤前"""
        transcript = """
        Operator: Good morning, and welcome to Apple's Fiscal Year 2024
        Fourth Quarter Earnings Conference Call. During today's presentation,
        all participants will be in a listen-only mode.
        """
        assert infer_earnings_session_from_transcript(transcript) == "BMO"

    def test_real_world_transcript_snippet_amc(self):
        """真實 transcript 片段測試 - 盤後"""
        transcript = """
        Operator: Good afternoon. My name is Connie, and I will be your
        conference operator today. At this time, I would like to welcome
        everyone to the NVIDIA Q3 Fiscal Year 2025 Earnings Conference Call.
        """
        assert infer_earnings_session_from_transcript(transcript) == "AMC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
