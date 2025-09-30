import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import How from "./components/How";
import Upload from "./components/Upload";
import Output from "./components/Output";
import Navbar from "./components/Navbar";
import { AnimatePresence } from "framer-motion";

const App = () => {
  return (
    <div>
      <Navbar />
      <AnimatePresence mode='wait'>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Home />} />
          <Route path="/how" element={<How />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/output" element={<Output />} />
        </Routes>
      </AnimatePresence>
    </div>
  );
};

export default App;
