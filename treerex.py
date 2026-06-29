import time
import logging

def main():
  return 1

if __name__ == "__main__":
  
    start = time.time()
    status = "completed"
    error_message = ""

    try:
        main()
    except Exception as e:
        status = "failed"
        error_message = f"{type(e).__name__}: {e}"
        raise
    finally:
        end = time.time()
        tot_time = end - start
        logger.info(f"\033[33mTotal runtime: {tot_time:.2f}s\033[0m")
        logger.info(f"\033[33mTotal runtime: {tot_time / 60:.2f}min\033[0m")
        logger.info(f"\033[33mTotal runtime: {tot_time / 3600:.2f}h\033[0m")
        append_run_summary(status, start, end, error_message)
