// Click a column header to sort high to low; click the same header again for low to high.
// Each cell's data-value holds the raw number; an empty data-value (no data) always sorts last.
document.querySelectorAll("table.sortable").forEach((table) => {
  const body = table.tBodies[0];
  // The last header row holds the column names (a row of group labels may sit above it)
  const headerRows = table.tHead.rows;
  const headers = Array.from(headerRows[headerRows.length - 1].cells);

  headers.forEach((header, column) => {
    header.querySelector("button").addEventListener("click", () => {
      const descending = header.getAttribute("aria-sort") !== "descending";
      headers.forEach((h) => h.removeAttribute("aria-sort"));
      header.setAttribute("aria-sort", descending ? "descending" : "ascending");

      const rows = Array.from(body.rows);
      rows.sort((a, b) => {
        const x = a.cells[column].dataset.value;
        const y = b.cells[column].dataset.value;
        if (x === "" || y === "") return (x === "") - (y === "");
        return descending ? Number(y) - Number(x) : Number(x) - Number(y);
      });
      body.append(...rows);
    });
  });
});
