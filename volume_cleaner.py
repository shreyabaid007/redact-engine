from modal import App, Period, Volume
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App("your-cleanup")
volume = Volume.from_name("your-volume-name")


@app.function(volumes={"/data": volume}, schedule=Period(hours=1))
def cleanup_old_files():
    """
    Deletes files older than 1 hour from the specified Modal volume.
    Logs a summary of the cleanup process, including the total space freed.
    """
    try:
        # Calculate the cutoff time
        cutoff_time = datetime.now() - timedelta(hours=1)
        logger.info(f"Starting cleanup for files older than {cutoff_time}")

        deleted_files = []
        retained_files = []

        # Iterate over files in the volume
        for filename in os.listdir("/data"):
            file_path = os.path.join("/data", filename)

            if os.path.isfile(file_path):  # Ensure the item is a file
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_mtime < cutoff_time:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)  # Delete the file
                        deleted_files.append((filename, file_size))
                    except Exception as e:
                        logger.error(f"Error deleting file {filename}: {e}")
                else:
                    retained_files.append(filename)

        # Log the results
        total_space_freed = sum(size for _, size in deleted_files)
        logger.info(f"Deleted {len(deleted_files)} files, freeing {total_space_freed / 1024 / 1024:.2f} MB")
        logger.info(f"Retained {len(retained_files)} files")

        # Summary output
        summary = {
            "timestamp": datetime.now().isoformat(),
            "cutoff_time": cutoff_time.isoformat(),
            "deleted_count": len(deleted_files),
            "retained_count": len(retained_files),
            "space_freed_mb": round(total_space_freed / 1024 / 1024, 2),
        }

        logger.info("Cleanup Summary:")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")

        return summary

    except Exception as e:
        logger.error(f"Cleanup process failed: {e}")
        raise


if __name__ == "__main__":
    app.run()