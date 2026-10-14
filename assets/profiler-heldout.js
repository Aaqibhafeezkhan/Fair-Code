/* ════════════════════════════════════════════════════════════════════════
   Fair Code - held-out column rows for the proxy check (issues #801-#803)

   Renders "file + column name" rows into a container (add/remove, any number
   of columns, .csv/.tsv/.json/.xlsx) and collects them into the specs that
   FairCodeProfiler.buildHeldOut() turns into a {column: values} map - the
   browser counterpart of repeating --proxy-hints-with PATH=COLUMN. Shared by
   the single-dataset and compare views. Column and join key are separate
   inputs (#822/#869), so a held-out column whose name contains a colon
   (e.g. race:self_reported) needs no escaping here - unlike the CLI's
   PATH=COLUMN:KEY string form.
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  // Spellings of the picker's encodings the CLI accepts after "@" (#924).
  var ENCODING_ALIASES = { 'auto': 'auto', 'utf-8': 'utf-8', 'utf8': 'utf-8', 'utf-16': 'utf-16', 'utf16': 'utf-16',
    'cp1252': 'cp1252', 'windows-1252': 'cp1252', 'latin-1': 'latin-1', 'latin1': 'latin-1', 'iso-8859-1': 'latin-1' };

  'use strict';

  var ACCEPT = '.csv,.tsv,.json,.xlsx,text/csv,application/json,' +
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

  // `getEncoding` (optional) returns the text-encoding choice of the page's picker (#857).
  function init(container, addBtn, label, getEncoding) {
    function addRow() {
      var row = document.createElement('div');
      row.className = 'heldout-row';
      var fileLabel = document.createElement('label');
      fileLabel.textContent = 'File ';
      var file = document.createElement('input');
      file.type = 'file';
      file.accept = ACCEPT;
      file.setAttribute('aria-label', label + ' file');
      fileLabel.appendChild(file);
      var colLabel = document.createElement('label');
      colLabel.textContent = 'Column ';
      var col = document.createElement('input');
      col.type = 'text';
      col.className = 'threshold-input';
      col.placeholder = 'e.g. race or race:self_reported';
      col.autocomplete = 'off';
      col.setAttribute('aria-label', label + ' column name');
      colLabel.appendChild(col);
      var keyLabel = document.createElement('label');
      keyLabel.textContent = 'Join key (optional) ';
      var key = document.createElement('input');
      key.type = 'text';
      key.className = 'threshold-input';
      key.placeholder = 'e.g. id';
      key.autocomplete = 'off';
      key.setAttribute('aria-label', label + ' join key column (optional)');
      keyLabel.appendChild(key);
      var normLabel = document.createElement('label');
      normLabel.textContent = 'Normalise key ';
      var norm = document.createElement('input');
      norm.type = 'checkbox';
      norm.setAttribute('aria-label', label + ' normalise join key (ignore case, spaces, leading zeros)');
      normLabel.appendChild(norm);
      var encodingLabel = document.createElement('label');
      encodingLabel.textContent = 'Encoding ';
      var encodingSelect = document.createElement('select');
      encodingSelect.className = 'threshold-input';
      encodingSelect.setAttribute('aria-label', label + ' text encoding');
      [
        ['inherit', 'Dataset setting'],
        ['auto', 'Auto (BOM, else UTF-8)'],
        ['utf-8', 'UTF-8'],
        ['utf-16', 'UTF-16'],
        ['cp1252', 'Windows-1252 (cp1252)'],
        ['latin-1', 'ISO-8859-1 (latin-1)']
      ].forEach(function (choice) {
        var option = document.createElement('option');
        option.value = choice[0];
        option.textContent = choice[1];
        encodingSelect.appendChild(option);
      });
      encodingSelect.value = 'inherit';
      encodingLabel.appendChild(encodingSelect);
      var remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'heldout-remove';
      remove.textContent = '✕';
      remove.setAttribute('aria-label', 'Remove this ' + label + ' column');
      remove.addEventListener('click', function () {
        if (container.children.length > 1) container.removeChild(row);
        else { file.value = ''; col.value = ''; key.value = ''; norm.checked = false; }
      });
      row.appendChild(fileLabel);
      row.appendChild(colLabel);
      row.appendChild(keyLabel);
      row.appendChild(normLabel);
      row.appendChild(encodingLabel);
      row.appendChild(remove);
      container.appendChild(row);
    }
    addRow();
    addBtn.addEventListener('click', addRow);

    // Resolves to [{name, column, key?, data, file}] for every row the user filled in;
    // throws if a row has only one of file/column. [] means "no held-out test".
    async function collect() {
      var specs = [];
      var rows = Array.prototype.slice.call(container.children);
      for (var i = 0; i < rows.length; i++) {
        var inputs = rows[i].querySelectorAll('input');
        var file = inputs[0].files && inputs[0].files[0];
        var column = inputs[1].value.trim();
        var key = inputs[2].value.trim();
        var normalize = !!inputs[3].checked;
        // The CLI's per-spec marker pasted into the key box: ~ normalises, ! keeps exact (#912).
        var marker = /[~!]$/.exec(key);
        if (marker && key.length > 1) {
          key = key.slice(0, -1).trim();
          normalize = marker[0] === '~';
          inputs[2].value = key;
          inputs[3].checked = normalize;
        }
        var rowEncoding = rows[i].querySelectorAll('select')[0];
        // The CLI's @ENCODING suffix pasted into the column box (race@latin-1) picks the row's
        // encoding; a trailing @word that is not a known encoding stays part of the column (#924).
        var suffix = /@([A-Za-z0-9_.-]+)$/.exec(column);
        if (suffix && rowEncoding && ENCODING_ALIASES[suffix[1].toLowerCase()]) {
          rowEncoding.value = ENCODING_ALIASES[suffix[1].toLowerCase()];
          column = column.slice(0, suffix.index).trim();
          inputs[1].value = column;
        }
        var encodingChoice = rowEncoding && rowEncoding.value !== 'inherit' ?
          rowEncoding.value : (getEncoding ? getEncoding() : 'auto');
        if (!file && !column) continue;
        if (!file || !column) throw new Error('each held-out row needs both a file and a column name');
        var data, encoding;
        var encodingIgnored = false;
        if (/\.xlsx$/i.test(file.name)) {
          data = await file.arrayBuffer();
          // A text encoding cannot apply to a workbook; buildHeldOut() says so (#952).
          encodingIgnored = !!encodingChoice && encodingChoice !== 'auto';
        }
        else if (getEncoding && window.FairCodeProfiler && window.FairCodeProfiler.decodeText) {
          // `encoding` is the explicit pick or the sniffed BOM (null for plain UTF-8), recorded in provenance (#891).
          var decoded = window.FairCodeProfiler.decodeText(await file.arrayBuffer(), encodingChoice);
          data = decoded.text;
          encoding = decoded.encoding || undefined;
        } else data = await file.text();
        specs.push({ name: file.name, column: column, key: key || undefined, normalize: (key && normalize) || undefined, encoding: encoding, encodingIgnored: encodingIgnored || undefined, data: data, file: file });
      }
      return specs;
    }

    function reset() {
      while (container.children.length > 0) {
        container.removeChild(container.children[0]);
      }
      container.innerHTML = '';
      addRow();
    }

    return { collect: collect, reset: reset };
  }


  window.FairCodeHeldOut = { init: init };
})();
