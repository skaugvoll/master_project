import pandas as pd 
import numpy as np




def csv_chunker( filename, chunk_size, columns=None, n_days=1, ts_index=0 ):
  '''
  Read csv in chunks, and only return rows within a datetime window
  Assumes that the first column contains timestamps
  '''
  # Open csv file
  it = pd.read_csv( filename, header='infer' if columns is None else None, chunksize=chunk_size, names=columns, parse_dates=[ts_index] )

  # Get first chunk
  chunk = next( it )

  # Get name for timestamp column
  ts_col = columns[ts_index] if not columns is None else chunk.columns[ts_index]


  # Extract first and first+nth day
  dt0 = chunk[ts_col].iloc[0]


  start_dt = pd.Timestamp( dt0.year, dt0.month, dt0.day ) + pd.Timedelta( days=1 )
  end_dt   = pd.Timestamp( dt0.year, dt0.month, dt0.day ) + pd.Timedelta( days=n_days+1 )

  # Scan to first valid timestamp
  while chunk[ts_col].iloc[-1] < start_dt:
    chunk = next( it )
  yield chunk[ np.logical_and(chunk[ts_col] >= start_dt, chunk[ts_col] < end_dt) ]

  # Then yield until last valid timestamp
  chunk = next( it )
  while True:
    if chunk[ts_col].iloc[-1] >= end_dt:
      chunk = chunk[ chunk[ts_col] < end_dt ]
      if len( chunk ) == 0: break
      yield chunk
      break
    yield chunk
    chunk = next( it )


def batch_iterator( dataframe_iterator, batch_size, allow_incomplete=True ):
  '''
  Take a dataframe iterator and partition its output
  into batches of specified size
  '''
  # TODO: Find a neater way around this. It prevents errors when setting a view
  pd.set_option('mode.chained_assignment', None)

  # Define helper variables
  bi = 0 # Index in current batch
  ci = 0 # Index in current chunk

  chunk = next( dataframe_iterator )
  batch = pd.DataFrame({ col: np.empty(batch_size, dtype=chunk[col].dtype ) for col in chunk.columns })

  while True:
    # Get amount to take from current chunk
    amount = min( len( chunk )-ci, batch_size-bi )

    # Copy over to batch dataframe
    for col in chunk.columns:
      batch[col][bi:bi+amount] = chunk[col].iloc[ci:ci+amount]

    # Get next chunk if current is exhausted
    if len( chunk )-ci < batch_size-bi:
      bi += amount # Increment batch index
      ci = 0 # Reset chunk index
      try:
        chunk = next( dataframe_iterator )
      except StopIteration:
        # Yield the last, partial batch if the chunk iterator is exhausted
        if allow_incomplete and bi > 0:
          yield batch[0:bi]
        break
    # Otherwise, prepare a new batch
    else:
      bi = 0
      ci += amount
      yield batch
      batch = pd.DataFrame({ col: np.empty(batch_size, dtype=chunk[col].dtype ) for col in chunk.columns })

