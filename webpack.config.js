const path = require("path");

const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: "./src/static/index.js",
  output: {
    filename: "main.js",
    path: path.resolve(__dirname, "dist")
  },

  // node requires `fs`, the browser doesn't. Adding this block of code
  // will prevent webpack from showing a warning. More on this link:
  // https://github.com/justadudewhohacks/face-api.js/issues/154#issuecomment-443420884
  node: {
    fs: "empty"
  },

  devServer: {
    contentBase: "./dist"
  },

  plugins: [
    // Copy static folder to build folder
    new CopyWebpackPlugin([
      {
        from: "./src/static/**/*.html",
        to: path.resolve(__dirname, "dist"),
        flatten: true
      },
      {
        from: "./src/static/**/*.mp4",
        to: path.resolve(__dirname, "dist/videos"),
        flatten: true
      },
      {
        from: "./ext/face-api.js/models/*",
        to: path.resolve(__dirname, "dist/models"),
        flatten: true
      },
      {
        from: "./src/main/model/*",
        to: path.resolve(__dirname, "dist/models"),
        flatten: true
      }
    ])
  ]
};
