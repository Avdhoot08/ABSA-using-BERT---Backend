import { parse } from "papaparse";

function parseCSVFile(file, numRows = 300) {
    return new Promise((resolve, reject) => {
        parse(file, {
            header: true,
            preview: numRows,
            dynamicTyping: true,
            complete: (results) => resolve(results.data),
            error: (error) => reject(error),
        });
    });
}

function getRandomElements(array, count = 1) {
    return array.sort(() => 0.5 - Math.random()).slice(0, count);
}

export { parseCSVFile, getRandomElements };
