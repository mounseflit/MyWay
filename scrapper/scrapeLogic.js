const puppeteer = require("puppeteer");
require("dotenv").config();

const scrapeLogic = async (url, mail, password, year, round) => {


  const browser = await puppeteer.launch({
    args: [
      "--disable-setuid-sandbox",
      "--no-sandbox",
      "--single-process",
      "--no-zygote",
    ],
    executablePath:
      process.env.NODE_ENV === "production"
        ? process.env.PUPPETEER_EXECUTABLE_PATH
        : puppeteer.executablePath(),
  });
  try {
    const page = await browser.newPage();

    await page.goto(url);


    let loggedIn = false;
    while (!loggedIn) {
        await page.waitForSelector(".item");
        await page.click(".item");

        await page.waitForSelector("#UserName");
        await page.type("#UserName", mail, { delay: 50 }); //mail

        await page.waitForSelector(".item");
        await page.click(".item");

        await page.waitForSelector("#Password");
        await page.type("#Password", password, { delay: 50 }); //password

        await page.waitForSelector(".item");
        await page.click(".item");

        await page.click("#btnSubmit");

        await page.goto("https://waliye.men.gov.ma/moutamadris/General/Home");

        try {
            await Promise.race([
                page.waitForNavigation({ timeout: 5000 }),
                page.waitForSelector("#NoteDiv", { timeout: 5000 }),
            ]);
            loggedIn = true;
        } catch (error) {
            console.error("Login attempt failed. Retrying...");
        }
    }

    await page.goto(
        "https://waliye.men.gov.ma/moutamadris/TuteurEleves/GetNotesEleve"
    );

    await page.waitForSelector("#SelectedAnnee");

    await page.select("#SelectedAnnee", year); // year

    await page.waitForSelector("#SelectedSession");

    await page.select("#SelectedSession", round); //round

    await page.click("#btnSearchNotes");

    await page.waitForSelector("#tab_notes_exam");

    await page.waitForSelector(
        "#ResultBulletin > div > div > div.widgetCont > div.nav-tabs-custom > ul > li:nth-child(2) > a",
        { delay: 5000 }
    );

    await page.click(
        "#ResultBulletin > div > div > div.widgetCont > div.nav-tabs-custom > ul > li:nth-child(2) > a"
    );

    const collectedData = [];

    for (let index = 1; index < 20; index++) {
        try {
            const [wl] = await page.$x(
                '//*[@id="tab_notes_exam"]/div[1]/div/table/tbody/tr[' +
                index +
                "]/td[1]",
                { delay: 100 }
            );
            if (!wl) {
                continue;
            }

            const sub = await wl.getProperty("textContent");
            const subject = await sub.jsonValue();

            const [wll] = await page.$x(
                '//*[@id="tab_notes_exam"]/div[1]/div/table/tbody/tr[' +
                index +
                "]/td[2]",
                { delay: 100 }
            );
            if (!wll) {
                continue;
            }

            const gra = await wll.getProperty("textContent");
            const grade = await gra.jsonValue();

            collectedData.push({ subject, grade });
        } catch (error) {
            console.error(`Error at index ${index}`);
        }
    }

    for (let index = 1; index < 3; index++) {
        try {
            const [wl] = await page.$x(
                '//*[@id="tab_notes_exam"]/div[2]/div[' + index + "]/label",
                { delay: 100 }
            );

            const sub = await wl.getProperty("textContent");
            const subject = await sub.jsonValue();

            const [wll] = await page.$x(
                '//*[@id="tab_notes_exam"]/div[2]/div[' + index + "]/span",
                { delay: 100 }
            );

            const gra = await wll.getProperty("textContent");
            const grade = await gra.jsonValue();

            collectedData.push({ subject, grade });
        } catch (error) {
            console.error(`Error at index ${index}`);
        }
    }



    await new Promise((resolve) => setTimeout(resolve, 5000));

    await page.goto("https://waliye.men.gov.ma/moutamadris/Account/LogOff");

 // Print the full title

    const jsonDataString = JSON.stringify(collectedData, null, 2);
    //console.log(jsonDataString);


    const jsonData = collectedData.map(entry => `${entry.subject.replace(/\s/g, '-')}:${entry.grade.replace(',', '.')}`).join('/');
    
   
    console.log(jsonData);
    res.send(jsonData);

  } catch (e) {
    console.error(e);
    res.send(`Something went wrong while running Puppeteer: ${e}`);
  } finally {
    await browser.close();
  }
};

module.exports = { scrapeLogic };

























// function rot13Encrypt(inputString) {
//     var result = '';
//     for (var i = 0; i < inputString.length; i++) {
//       var charCode = inputString.charCodeAt(i);
  
//       if (65 <= charCode && charCode <= 90) {
//         result += String.fromCharCode(((charCode - 65 + 13) % 26) + 65);
//       }
      
//       else if (97 <= charCode && charCode <= 122) {
//         result += String.fromCharCode(((charCode - 97 + 13) % 26) + 97);
//       }

//       else {
//         result += inputString.charAt(i);
//       }
//     }
//     return result;
//   }
  


//   rot13Encrypt("fx-duOeHipmm8J1RSp6fwBqG3OyoxSWw4d9TtHwu0pWdkvW91JN");












// let mail = "R130001518@taalim.ma";
// let password = "130569Akram";
// let year = "2022";
// let round = "2";
// scrapeProduct(
//     "https://waliye.men.gov.ma/moutamadris/Account",
//     mail,
//     password,
//     year,
//     round
// );






