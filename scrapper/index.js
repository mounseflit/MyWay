const express = require("express");
const { scrapeLogic } = require("./scrapeLogic");
const app = express();

const PORT = process.env.PORT || 4000;

app.get("/scrape", (req, res) => {
  try {
    let url = "https://waliye.men.gov.ma/moutamadris/Account";
    let mail = "R130001518@taalim.ma";
    let password = "130569Akram";
    let year = "2022";
    let round = "2";
    const message = scrapeLogic(url, mail, password, year, round);
    res.json({ message });
  } catch {
    res.status(500).json({ error: "Failed to scrape data" });
  }
});

app.get("/", (req, res) => {
  res.send("Render Puppeteer server is up and running!");
});

app.listen(PORT, () => {
  console.log(`Listening on port ${PORT}`);
});
