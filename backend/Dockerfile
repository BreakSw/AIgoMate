FROM maven:3.9.9-eclipse-temurin-21 AS build

WORKDIR /workspace
COPY backend/pom.xml ./pom.xml
RUN mvn --batch-mode --no-transfer-progress dependency:go-offline

COPY backend/src ./src
RUN mvn --batch-mode --no-transfer-progress -DskipTests package

FROM eclipse-temurin:21-jre-jammy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home algomate

WORKDIR /app
RUN mkdir -p /app/data && chown -R algomate:algomate /app

COPY --from=build --chown=algomate:algomate /workspace/target/*.jar /app/algomate-backend.jar

USER algomate

EXPOSE 8898

ENTRYPOINT ["java", "-jar", "/app/algomate-backend.jar"]
