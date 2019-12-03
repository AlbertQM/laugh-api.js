const path = require("path");

const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: "./src/static/index.js",
  output: {
    filename: "main.js",
    path: path.resolve(__dirname, "dist")
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
        from: "./ext/face-api.js/models/*",
        to: path.resolve(__dirname, "dist/models"),
        flatten: true
      }
    ])
  ]
};
