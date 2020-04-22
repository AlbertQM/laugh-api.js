const fs = require("fs-extra");
const BASE_DIR = "./annotations/";
fs.readdir(BASE_DIR)
  .then(fileList => {
    fileList.forEach(file => {
      const contents = fs.readFileSync(`${BASE_DIR}${file}`, {
        encoding: "utf8"
      });
      const rows = contents.split("\n");
      let sanitisedData = "";
      rows.forEach(row => {
        if (row.length === 0) {
          return;
        }
        // Take only the first 5 columns
        // Filename _ Filename Start End
        // BillGross_2003 1 BillGross_2003 1002.47 1011.51 <o,f0,female> really good combination of those two factors {BREATH} it turns out there's a lot of powerful sun {BREATH} all around the world obviously but in special places(2) {NOISE} where it happens to be relatively inexpensive to place these <sil> (BillGross_2003-1002.47-1011.51-F0_F-S135)
        sanitisedData += row.split("<")[0].trimRight() + "\n";
      });
      fs.writeFileSync(`${BASE_DIR}${file}`, sanitisedData);
    });
  })
  .catch(e => {
    console.error("Whoooops, ", e);
  });
