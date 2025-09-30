// context/PredictionContext.js
import { createContext, useState, useContext } from "react";

const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {
  const [resumeFile, setResumeFile] = useState(null);
  const [fit, setFit] = useState(null);

  return (
    <PredictionContext.Provider value={{ resumeFile, setResumeFile, fit, setFit }}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = () => useContext(PredictionContext);
