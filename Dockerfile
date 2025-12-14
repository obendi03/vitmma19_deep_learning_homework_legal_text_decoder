FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Work directory
WORKDIR /app

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code and run script
COPY ./src .

# 5. Make run.sh executable
RUN chmod +x run.sh

# 6. Default command
CMD ["bash", "run.sh"]
