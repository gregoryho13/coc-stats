// import Chart from 'chart.js/auto';
// import nlp from "compromise";

function injectHTML(text) {
  console.log("fired injecthtml");
  const target = document.querySelector("#lyrics");
  target.innerHTML = text;
}

function runNLP(text) {
  console.log(text.replace(/\n/g, " "));

  let doc = nlp(text.replace(/\n/g, " "));

  console.log("nlp on text");

  // Topics
  let topicList = doc.clauses();
  topicList.normalize();

  console.log(topicList);

  topicList = topicList.out('array');
  let topicOccurrence = [];
  topicList.forEach((element) => {
    const regexp = new RegExp("\\b" + element + "\\b", "g");
    topicOccurrence.push(doc.text().match(regexp));
  });

  console.log("topic list", topicList);
  console.log("topic occurrences", topicOccurrence);

  const ctx = document.getElementById('chart');

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: topicList,
      datasets: [
        {
          label: "# of Occurrences",
          data: topicOccurrence,
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

async function mainEvent() {
  const mainForm = document.querySelector(".main_form");
  const songTextField = document.querySelector("#song_input");
  const artistTextField = document.querySelector("#artist_input");
  const submitDataButton = document.querySelector("#submit");

  console.log("main");

  submitDataButton.addEventListener("click", async (submitEvent) => {
    console.log("Submitting data");

    const songName = document.querySelector("#song_input").value;
    const artistName = document.querySelector("#artist_input").value;

    console.log("song", songName);
    console.log("artist", artistName);

    const params = {
      song: songName,
      artist: artistName,
    };

    const options = {
      method: "GET",
    };

    const results = await fetch(
        'http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect' + '?' + new URLSearchParams(params),
        options,
    );
    console.log(results);

    const xmlData = await results.text();
    console.log(xmlData);

    const XMLparser = new DOMParser();
    const doc = XMLparser.parseFromString(xmlData, "text/xml");

    const lyrics = doc.getElementsByTagName("Lyric")[0].childNodes[0].nodeValue;

    // Testing
    // lyrics = await fetch("sample.txt").then((response) => response.text());

    console.log(lyrics);
    injectHTML(lyrics);

    runNLP(lyrics);
  });
}

document.addEventListener("DOMContentLoaded", async () => mainEvent());
