from modal import App, Period, Volume
import os
from datetime import datetime, timedelta

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App and Volume Setup
app = App("your-cleanup")
volume = Volume.from_name("your-volume-name")


@app.function(volumes={"/data": volume}, schedule=Period(hours=1))
def cleanup_old_files(retention_hours: int = 1, target_dir: str = "/data"):
    """
    Deletes files in the specified directory that are older than the given retention period.

    Args:
        retention_hours (int): The retention period in hours. Files older than this will be deleted.
        target_dir (str): The directory to clean up.

    Returns:
        dict: Summary of the cleanup process, including the number of deleted and retained files,
              total space freed, and any errors encountered.
    """
    try:
        # Calculate the cutoff time
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        deleted_files = []
        retained_files = []
        error_files = []

        # Ensure the target directory exists
        if not os.path.exists(target_dir):
            return {"status": "no_target_dir", "target_dir": target_dir}

        # Iterate over files in the directory
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)

            try:
                if os.path.isfile(file_path):  # Process only files
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                    if file_mtime < cutoff_time:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)  # Delete the file
                            deleted_files.append((filename, file_size))
                        except Exception as deletion_error:
                            error_files.append((filename, str(deletion_error)))
                    else:
                        retained_files.append(filename)
            except Exception as processing_error:
                error_files.append((filename, str(processing_error)))

        # Calculate the total space freed
        total_space_freed = sum(size for _, size in deleted_files)

        # Return a summary
        return {
            "timestamp": datetime.now().isoformat(),
            "cutoff_time": cutoff_time.isoformat(),
            "deleted_count": len(deleted_files),
            "retained_count": len(retained_files),
            "space_freed_mb": round(total_space_freed / 1024 / 1024, 2),
            "error_count": len(error_files),
        }

    except Exception as e:
        raise RuntimeError(f"Cleanup process failed: {e}")


if __name__ == "__main__":
    # Run the cleanup with custom retention and target directory for experimentation
    test_retention_hours = 2  # Delete files older than 2 hours
    test_target_dir = "/data"  # Target directory for cleanup
    result = cleanup_old_files(test_retention_hours, test_target_dir)
    print("Cleanup Result:", result)
