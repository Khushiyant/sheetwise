import pytest
import pandas as pd
from sheetwise.privacy import PIIRedactor
from sheetwise.core import SpreadsheetLLM

@pytest.fixture
def pii_dataframe():
    data = {
        "Name": ["John Doe", "Jane Smith", "Alice Jones"],
        "Email": ["john.doe@example.com", "jane.smith@work.org", "alice@gmail.com"],
        "Phone": ["123-456-7890", "(987) 654-3210", "+1 555 123 4567"],
        "CreditCard": ["1234-5678-9012-3456", "1111 2222 3333 4444", "safe value"],
        "Notes": ["Contact me at john.doe@example.com", "IP is 192.168.1.1", "No PII here"]
    }
    return pd.DataFrame(data)

def test_redactor_basic(pii_dataframe):
    redactor = PIIRedactor()
    redacted_df = redactor.redact(pii_dataframe)
    
    # Check Emails
    assert redacted_df["Email"].str.contains(r"\[REDACTED\]").all()
    assert not redacted_df["Email"].str.contains("@").any()
    
    # Check Phone
    assert redacted_df["Phone"].str.contains(r"\[REDACTED\]").all()
    
    # Check Credit Card
    assert redacted_df["CreditCard"].iloc[0] == "[REDACTED]"
    assert redacted_df["CreditCard"].iloc[1] == "[REDACTED]"
    assert redacted_df["CreditCard"].iloc[2] == "safe value"

def test_redactor_custom_mask(pii_dataframe):
    redactor = PIIRedactor(mask="***")
    redacted_df = redactor.redact(pii_dataframe)
    
    assert redacted_df["Email"].str.contains(r"\*\*\*").all()

def test_redactor_notes_extraction(pii_dataframe):
    redactor = PIIRedactor()
    redacted_df = redactor.redact(pii_dataframe)
    
    # Email inside text
    assert "Contact me at [REDACTED]" in redacted_df["Notes"].iloc[0]
    
    # IP Address inside text
    assert "IP is [REDACTED]" in redacted_df["Notes"].iloc[1]

def test_llm_integration(pii_dataframe):
    llm = SpreadsheetLLM(enable_pii_redaction=True)
    
    # Compress and encode
    encoded = llm.compress_and_encode_for_llm(pii_dataframe)
    
    # Verify PII is gone from the output string
    assert "john.doe@example.com" not in encoded
    assert "123-456-7890" not in encoded
    assert "[REDACTED]" in encoded

def test_llm_integration_disabled(pii_dataframe):
    llm = SpreadsheetLLM(enable_pii_redaction=False)
    
    encoded = llm.compress_and_encode_for_llm(pii_dataframe)
    
    # Verify PII is present
    assert "john.doe@example.com" in encoded
