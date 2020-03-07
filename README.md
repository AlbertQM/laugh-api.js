# LOLd.js - Multimodal Lightweight Online Laughter Detection

```
* * * * * * * * * * * * * * * * * * * * * * * * *
* Final Year Project (2019/2020)                *
* BSc(Eng) "Creative Computing"                 *
* Queen Mary University of London               *
*                                               *
* @Authors Alberto Morabito                     *
* * * * * * * * * * * * * * * * * * * * * * * * *
```

JavaScript (TS) API for laughter recognition with audiovisual data, built using Tensorflow.js <br>

The multimodal recognition uses two different models:

- For video data the model used is [face-api.js](https://github.com/justadudewhohacks/face-api.js/)
- For audio data, a custom model has been built (using RNN with LSTMs)

## Setup dev environment

1. we need to transpile the `.js` code to `.ts`: <br>
   `tsc --watch` <br>
   Using the watch flag so that the files are transpiled automatically upon change.

2. we need to serve the files: <br>
   `npm start` <br>
   This will have `webpack-dev-server` serve the files on `localhost:8080`
