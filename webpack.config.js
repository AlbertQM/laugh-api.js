const path = require("path");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const { WebpackPluginServe: Serve } = require("webpack-plugin-serve");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  context: __dirname,
  mode: "development",
  entry: ["./src/index.ts", "webpack-plugin-serve/client"],
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"]
  },
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "dist")
  },

  plugins: [
    // Clean up the dist folder
    new CleanWebpackPlugin(),

    // Copy static folder to build folder
    new CopyWebpackPlugin([
      {
        from: "./src/static/**/*",
        to: path.resolve(__dirname, "dist"),
        flatten: true
      },
      {
        from: "./ext/face-api.js/models/*",
        to: path.resolve(__dirname, "dist/models"),
        flatten: true
      }
    ]),

    // Serve static files for dev
    new Serve({
      static: [path.resolve(__dirname, "dist/")]
    })
  ],

  watch: true,

  // Simplify debugging by having actual filenames in console errors
  devtool: "inline-source-map"
};
