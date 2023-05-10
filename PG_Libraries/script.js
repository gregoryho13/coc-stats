/*
  Hook this script to index.html
  by adding `<script src="script.js">` just before your closing `</body>` tag
*/

function getRandomIntInclusive(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function injectHTML(list) {
  console.log("fired injectHTML");
  const target = document.querySelector("#library_list");
  target.innerHTML = "";
  list.forEach((item) => {
    const str = `<li>${item.branch_name} (${item.branch_type})</li>`;
    target.innerHTML += str;
  });
}

/* A quick filter that will return something based on a matching input */
function filterList(list, query) {
  return list.filter((item) => {
    const lowerCaseName = item.branch_name.toLowerCase();
    const lowerCaseQuery = query.toLowerCase();
    return lowerCaseName.includes(lowerCaseQuery);
  });
}

function cutLibraryList(list) {
  console.log("fired cut list");
  const range = [...Array(list.length > 15 ? 15 : list.length).keys()];
  const bucketIndices = [];
  return (newArray = range.map((item) => {
    let index = getRandomIntInclusive(0, list.length - 1);
    while (bucketIndices.includes(index)) {
      index = getRandomIntInclusive(0, list.length - 1);
    }
    bucketIndices.push(index);
    return list[index];
  }));
}

function initMap() {
  const carto = L.map('map').setView([38.990, -76.938], 13);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }).addTo(carto);
  return carto;
}

function markerPlace(array, map) {
  console.log('array for markers', array);

  map.eachLayer((layer) => {
    if (layer instanceof L.Marker) {
      layer.remove();
    }
  });

  array.forEach((item) => {
    console.log('markerPlace', item);
    const coordinates = item.location_1;

    L.marker([coordinates.latitude, coordinates.longitude], {title: (item.branch_name + ' (' + item.branch_type + ')')}).addTo(map);
  });
}

function displayChart(array, chart) {
  let data_list = {};
  array.forEach((item) => {
    const branch_type = item.branch_type
    if (branch_type in data_list) {
      data_list[branch_type] += 1;
    } else {
      data_list[branch_type] = 1;
    }
  });

  console.log(data_list);

  const chartStatus = Chart.getChart("libraryChart");
  if (chartStatus != undefined) {
    chartStatus.destroy();
  }

  new Chart(chart, {
    type: 'bar',
    data: {
      labels: Object.keys(data_list),
      datasets: [{
        label: '# of type',
        data: Object.values(data_list),
        borderWidth: 1
      }]
    },
    options: {
      response: true,
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

async function mainEvent() {
  // the async keyword means we can make API requests
  const mainForm = document.querySelector(".main_form"); // This class name needs to be set on your form before you can listen for an event on it
  const loadDataButton = document.querySelector("#data_load");
  const clearDataButton = document.querySelector("#data_clear");
  const generateListButton = document.querySelector("#generate");
  const textField = document.querySelector("#resto");
  const chartArea = document.querySelector(".chart_area");

  const loadAnimation = document.querySelector("#data_load_animation");
  loadAnimation.style.display = "none";
  generateListButton.classList.add("hidden");

  const carto = initMap();
  const ctx = document.getElementById('libraryChart');
  chartArea.classList.add("hidden");

  const storedData = localStorage.getItem("storedData");
  let parsedData = JSON.parse(storedData);
  if (parsedData?.length > 0) {
    generateListButton.classList.remove("hidden");
  }

  let currentList = []; // this is "scoped" to the main event function

  /* We need to listen to an "event" to have something happen in our page - here we're listening for a "submit" */
  loadDataButton.addEventListener("click", async (submitEvent) => {
    // async has to be declared on every function that needs to "await" something
    console.log("Loading data");
    loadAnimation.style.display = "inline-block";

    // Basic GET request - this replaces the form Action
    const results = await fetch(
      "https://data.princegeorgescountymd.gov/resource/7k64-tdwr.json"
    );

    // This changes the response from the GET into data we can use - an "object"
    const storedList = await results.json();
    console.log(storedList);
    localStorage.setItem("storedData", JSON.stringify(storedList));
    parsedData = storedList;

    if (storedList?.length > 0) {
      generateListButton.classList.remove("hidden");
    }

    loadAnimation.style.display = "none";
    console.table(storedList);
  });

  generateListButton.addEventListener("click", (event) => {
    console.log("generate new list");
    
    currentList = cutLibraryList(parsedData);
    console.log(currentList);
    injectHTML(currentList);
    markerPlace(currentList, carto);
    chartArea.classList.remove("hidden");
    displayChart(currentList, ctx);
  });
  
  textField.addEventListener("input", (event) => {
    console.log("input", event.target.value);
    const newList = filterList(currentList, event.target.value);
    console.log(newList);
    injectHTML(newList);
    markerPlace(newList, carto);
    chartArea.classList.remove("hidden");
    displayChart(newList, ctx);
  });
  
  clearDataButton.addEventListener("click", (event) => {
    console.log("clear browser data");
    localStorage.clear();
    console.log('localStorage Check', localStorage.getItem("storedData"))
  });
}

document.addEventListener("DOMContentLoaded", async () => mainEvent()); // the async keyword means we can make API requests
