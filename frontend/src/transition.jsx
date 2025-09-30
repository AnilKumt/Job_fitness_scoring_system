// transition.js
import React from "react";
import { motion } from "framer-motion";

const transition = (OgComponent) => {
  return function TransitionWrapper(props) {
    const transitionSettings = {
      duration: 0.9,
      ease: [0.76, 0, 0.24, 1], 
    };

    return (
      <>
       
        <OgComponent {...props} />

        
        <motion.div
          className="transition-layer layer-top"
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 0 }}
          exit={{ scaleY: 1 }}
          transition={transitionSettings}
        />
        <motion.div
          className="transition-layer layer-bottom"
          initial={{ scaleY: 1 }}
          animate={{ scaleY: 0 }}
          exit={{ scaleY: 0 }}
          transition={{ ...transitionSettings, delay: 0.2 }}
        />
      </>
    );
  };
};

export default transition;
