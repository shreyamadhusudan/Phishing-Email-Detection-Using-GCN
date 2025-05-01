import { useState } from "react";

export default function EmailClassifier() {
  const [emailText, setEmailText] = useState("");
  const [result, setResult] = useState(null);

  const classifyEmail = async () => {
    if (!emailText.trim()) {
      alert("Please enter an email text.");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: emailText }),
      });

      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error classifying email.");
    }
  };

  return (
    <div className="container">
      <h1 className="title">Email Phishing Classifier</h1>
      <textarea
        className="input-box"
        value={emailText}
        onChange={(e) => setEmailText(e.target.value)}
        placeholder="Paste email content here..."
      ></textarea>
      <button onClick={classifyEmail} className="button">
        Classify Email
      </button>
      {result && <p className={`prediction ${result ? "show" : ""}`}>Prediction: {result}</p>}
    </div>
  );
}