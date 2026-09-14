import { useRef, useState } from 'react';
import { FileSpreadsheet, Upload } from 'lucide-react';
import { useDataset } from '@/hooks/useDataset';
import { Card, cx } from '@/components/primitives';

// Feature counts are fixed by each dataset's frozen preprocessing
// pipeline (see api/config.py's Scope definitions) -- not something the
// UI derives, just states for the upload contract.
const NUM_FEATURES: Record<string, number> = { cicids2017: 70, nbaiot: 115 };

export default function CsvUpload({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
}) {
  const { dataset } = useDataset();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <Card>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) onFile(dropped);
        }}
        onClick={() => inputRef.current?.click()}
        className={cx(
          'cursor-pointer rounded-card border-2 border-dashed p-8 text-center transition-colors',
          dragging ? 'border-accent bg-accent-soft/40' : 'border-border hover:bg-surface-alt/60',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="flex flex-col items-center gap-1.5">
            <FileSpreadsheet size={26} className="text-accent" />
            <p className="text-sm font-medium">{file.name}</p>
            <p className="text-xs text-ink-muted">{(file.size / 1024).toFixed(1)} KB — click to replace</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1.5">
            <Upload size={26} className="text-ink-muted/60" />
            <p className="text-sm font-medium">Drop a preprocessed CSV, or click to browse</p>
            <p className="text-xs text-ink-muted">
              10 rows × {NUM_FEATURES[dataset] ?? '…'} columns, already run through this project's
              own preprocessing pipeline
            </p>
          </div>
        )}
      </div>
      <p className="mt-3 text-xs text-ink-muted">
        This must already be preprocessed (scaled features, correct column order) — the model does not
        accept raw packet captures or unprocessed flow logs.
      </p>
    </Card>
  );
}
