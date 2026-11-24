# Dev environment for the Spark Data Agent revamp.
#
# Linux container with Python 3.11 and JDK 17: Spark runs here, not on the
# Windows host, which is the whole point (no winutils, no version roulette).
#
# Pinned to bookworm on purpose: python:3.11-slim now resolves to trixie, whose
# only JDK is 21, and Spark 3.5 refuses to run on Java 21.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    # Bind the driver to loopback: in a container the hostname rarely resolves.
    SPARK_LOCAL_IP=127.0.0.1 \
    # pandas-on-Spark warns unless this is set before any SparkContext exists.
    PYARROW_IGNORE_TIMEZONE=1 \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are read from pyproject.toml instead of being copied here: one
# source of truth, so a version bump cannot drift from the image.
WORKDIR /opt/build
COPY pyproject.toml ./
RUN python - <<'PY'
import subprocess
import sys
import tomllib

project = tomllib.load(open("pyproject.toml", "rb"))["project"]
deps = project["dependencies"] + project["optional-dependencies"]["dev"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY

# Preload spark-xml and Hadoop S3A jars straight into the PySpark jars directory
# so the classpath is complete offline without `spark.jars.packages` (which would
# hit Maven at every cold start).
ARG SPARK_XML_VERSION=0.18.0
RUN cd "$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')/jars" \
    && curl -fsSLO "https://repo1.maven.org/maven2/com/databricks/spark-xml_2.12/${SPARK_XML_VERSION}/spark-xml_2.12-${SPARK_XML_VERSION}.jar"

RUN cd "$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')/jars" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"

# The repository is bind-mounted here at runtime (see docker-compose.yml), so
# the source is edited on the host and executed in the container.
WORKDIR /workspace

CMD ["sleep", "infinity"]
