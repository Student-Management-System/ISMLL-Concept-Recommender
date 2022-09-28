pipeline {
    agent any

    environment {
        DEMO_SERVER = '147.172.178.30'
    }

    stages {

        stage('Git') {
            steps {
                cleanWs()
                git branch: 'main', url: 'https://github.com/Student-Management-System/ISMLL-Concept-Recommender.git'
            }
        }

        stage('Build') {
            steps {
                script {
                    // Based on:
                    // - https://e.printstacktrace.blog/jenkins-pipeline-environment-variables-the-definitive-guide/
                    // - https://stackoverflow.com/a/16817748
                    // - https://stackoverflow.com/a/51991389
                    env.API_VERSION = sh(returnStdout: true, script: 'grep -Po "(?<=app = flask_app, version=\')[^\']+" Webservice.py').trim()
                    echo "API: ${env.API_VERSION}"
                    dockerImage = docker.build 'e-learning-by-sse/qualityplus-ismll-recommender'
                    docker.withRegistry('https://ghcr.io', 'github-ssejenkins') {
                        dockerImage.push("${env.API_VERSION}")
                        dockerImage.push('latest')
                    }
                }
            }
        }

        // Based on: https://medium.com/@mosheezderman/c51581cc783c
        stage('Deploy') {
            steps {
                sshagent(credentials: ['Stu-Mgmt_Demo-System']) {
                    sh """
                        # [ -d ~/.ssh ] || mkdir ~/.ssh && chmod 0700 ~/.ssh
                        # ssh-keyscan -t rsa,dsa example.com >> ~/.ssh/known_hosts
                        ssh -i ~/.ssh/id_rsa_student_mgmt_backend elscha@${env.DEMO_SERVER} <<EOF
                            cd ~/ISMLL-Concept-Recommender
                            git reset --hard
                            git pull
                            chmod +x demo.sh
                            systemctl --user restart recommender_api.service
                            exit
                        EOF"""
                }
            }
        }
    }
}
